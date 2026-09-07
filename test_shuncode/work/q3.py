# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8')
import statsmodels.api as sm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings; warnings.filterwarnings('ignore')
out={}
d=pd.read_csv('results/male_tidy.csv',encoding='utf-8')
d=d.dropna(subset=['y','week','bmi','age','height','weight']); d=d[d['y']>0]
d['logy']=np.log(d['y'])
gm=d.groupby('code')
d['dm_logy']=d['logy']-gm['logy'].transform('mean'); d['dm_week']=d['week']-gm['week'].transform('mean')
Xw=sm.add_constant(d[['dm_week']]); fe=sm.OLS(d['dm_logy'],Xw).fit(); b_w=float(fe.params['dm_week'])
rows=[]
for code,sub in gm:
    a=sub['logy'].mean()-b_w*sub['week'].mean(); t=(np.log(0.04)-a)/b_w
    rows.append({'code':code,'bmi':sub['bmi'].iloc[0],'age':sub['age'].iloc[0],
        'height':sub['height'].iloc[0],'weight':sub['weight'].iloc[0],'tstar':t,'a':a})
pw=pd.DataFrame(rows); pw['tstar']=pw['tstar'].clip(10,28)
X=sm.add_constant(pw[['bmi','age','height','weight']]); lf=sm.OLS(pw['tstar'],X).fit()
out['multifactor_tstar']={'params':{k:round(float(v),4) for k,v in lf.params.items()},
   'pvals':{k:('%g'%v) for k,v in lf.pvalues.items()},'rsq':round(float(lf.rsquared),4),
   'adj':round(float(lf.rsquared_adj),4),'f':round(float(lf.fvalue),3),'f_p':('%g'%lf.f_pvalue)}
feat=pw[['bmi']].values; sc=StandardScaler().fit(feat); Z=sc.transform(feat)
sil={}
for K in range(2,7):
    km=KMeans(n_clusters=K,n_init=20,random_state=42).fit(Z); sil[K]=round(float(silhouette_score(Z,km.labels_)),4)
out['bmi_silhouette']=sil
bestK=int(max(sil,key=sil.get)); km=KMeans(n_clusters=bestK,n_init=20,random_state=42).fit(Z)
centers=sc.inverse_transform(km.cluster_centers_).flatten()
out['bmi_kmeans_K']=bestK
grp_info={}
for lbl in sorted(set(km.labels_)):
    sub=pw[km.labels_==lbl]
    grp_info[str(int(lbl))]={'n':int(len(sub)),'bmi_min':round(float(sub['bmi'].min()),2),
      'bmi_max':round(float(sub['bmi'].max()),2),'bmi_mean':round(float(sub['bmi'].mean()),2),
      'center':round(float(centers[lbl]),2),'tstar_median':round(float(sub['tstar'].median()),2)}
out['bmi_kmeans_groups']=grp_info
pw['grp']=[int(np.argmin(np.abs(centers-b))) for b in pw['bmi']]
sigma_e=float(np.sqrt((d['dm_logy']-fe.fittedvalues).var())); out['sigma_e']=round(sigma_e,4)
def reach(sub,sig,w):
    yhat=np.exp(sub['a'].values+b_w*w)
    return float(np.mean([1-stats.norm.cdf((np.log(0.04)-np.log(yi))/sig) for yi in yhat]))
# graded risk (tiered) from problem
def rk(w):
    return 1 if w<=12 else (3 if w<=27 else 6)
def solve(sub,sig,alpha=0.90):
    # earliest week with noise-aware reach>=alpha
    curve={}; best=None
    for w in np.arange(11,28.01,0.5):
        A=reach(sub,sig,w); curve[round(float(w),1)]=round(A,4)
        if A>=alpha and best is None: best=(w,A)
    if best is None:
        # fallback: last week
        best=(28.0,round(curve[28.0],4))
    return {'week':round(float(best[0]),1),'reach':round(float(best[1]),4),
            'risk_level':rk(best[0]),'curve':curve}
rg={}
for lbl in sorted(pw['grp'].unique()):
    sub=pw[pw['grp']==lbl]
    if len(sub)<3: continue
    rr=solve(sub,sigma_e)
    rg[str(int(lbl))]={'n':int(len(sub)),'bmi_mean':round(float(sub['bmi'].mean()),2),
       'best_week':rr['week'],'reach':rr['reach'],'risk_level':rr['risk_level']}
out['Q3_groups_alpha90']=rg
rg95={}
for lbl in sorted(pw['grp'].unique()):
    sub=pw[pw['grp']==lbl]
    if len(sub)<3: continue
    rr=solve(sub,sigma_e,0.95)
    rg95[str(int(lbl))]={'n':int(len(sub)),'best_week':rr['week'],'reach':rr['reach']}
out['Q3_groups_alpha95']=rg95
# MC: sensitivity of best week to sigma (detection error magnitude)
mc={}
for sig in [0.12,0.15,0.20,0.2145,0.30,0.40]:
    rr={}
    for lbl in sorted(pw['grp'].unique()):
        sub=pw[pw['grp']==lbl]
        if len(sub)<3: continue
        rr[str(int(lbl))]=solve(sub,sig)['week']
    mc[str(sig)]=rr
out['Q3_mc_sigma']=mc
lab=int(pw['grp'].value_counts().idxmax()); sub=pw[pw['grp']==lab]
out['Q3_fig_curve_group']=str(int(lab)); out['Q3_fig_curve']=solve(sub,sigma_e)['curve']
out['Q3_reach_bylabel']={str(int(lb)):solve(sub,sigma_e)['curve'] for lb in sorted(pw['grp'].unique()) for sub in [pw[pw['grp']==lb]] if len(sub)>=3}
with open('results/q3.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,default=str)
print('Q3_DONE bestK',bestK,'sigma_e',round(sigma_e,4))

