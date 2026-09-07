# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8')
import statsmodels.api as sm
import warnings; warnings.filterwarnings('ignore')
out={}
d=pd.read_csv('results/male_tidy.csv',encoding='utf-8')
d=d.dropna(subset=['y','week','bmi']); d=d[d['y']>0]
d['logy']=np.log(d['y'])
gm=d.groupby('code')
d['dm_logy']=d['logy']-gm['logy'].transform('mean')
d['dm_week']=d['week']-gm['week'].transform('mean')
Xw=sm.add_constant(d[['dm_week']]); fe=sm.OLS(d['dm_logy'],Xw).fit()
b_w=float(fe.params['dm_week'])
# residual std of within-woman concentration (measurement/lab noise proxy)
resid=d['dm_logy']-fe.fittedvalues
sigma_e=float(resid.std())
out['within_slope']=round(b_w,5); out['resid_sd_log']=round(sigma_e,4)
# per-woman threshold week t* (fitted within model), clipped to [10,28]
rows=[]
for code,sub in gm:
    logy_m=sub['logy'].mean(); week_m=sub['week'].mean()
    a=logy_m-b_w*week_m
    t=(np.log(0.04)-a)/b_w
    rows.append({'code':code,'bmi':sub['bmi'].iloc[0],'age':sub['age'].iloc[0],
        'height':sub['height'].iloc[0],'weight':sub['weight'].iloc[0],'t':t})
pw=pd.DataFrame(rows)
pw['tstar']=pw['t'].clip(10,28)
# graded detection-time risk per problem: <=12 low(1),13-27 high(3),>=28 very high(6)
def rk(w):
    if w<=12: return 1
    elif w<=27: return 3
    else: return 6
out['graded_risk']={'<=12w':1,'13-27w':3,'>=28w':6}
def solve_grouped(sub, alpha):
    ts=sub['tstar']
    # reliability-target weeks
    wk=[]
    # earliest week meeting alpha coverage
    best=None
    for w in range(10,29):
        A=float(np.mean(ts<=w))
        if A>=alpha:
            if best is None: best=(w,A)
    if best is None: best=(28,float(np.mean(ts<=28)))
    return {'alpha':alpha,'week':float(best[0]),'reach':round(float(best[1]),4),
            'risk_level':rk(best[0])}
groups={'G1[20,28)':(20,28),'G2[28,32)':(28,32),'G3[32,36)':(32,36),'G4[36,40)':(36,40),'G5[40+)':(40,120)}
res95={}; res89={}
for name,(lo,hi) in groups.items():
    sub=pw[(pw['bmi']>=lo)&(pw['bmi']<hi)]
    if len(sub)==0: continue
    res95[name]={'n':int(len(sub)),'bmi_range':[round(float(sub['bmi'].min()),2),round(float(sub['bmi'].max()),2)],
        'bmi_mean':round(float(sub['bmi'].mean()),2),**solve_grouped(sub,0.90),
        'mean_tstar':round(float(sub['tstar'].mean()),2)}
    res89[name]={'n':int(len(sub)),**solve_grouped(sub,0.95)}
out['Q2_alpha90']=res95
out['Q2_alpha95']=res89
# ---- Optimal GROUPS (Q3 requires reasonable grouping). Use KMeans or optimal partition? ----
# We'll also apply data-driven optimization: search group split points on BMI to minimize global risk.
# For Q3 we keep the 5 fixed for comparability but add factors via clustering and error MC.
# ---- reach curves for plotting ----
reach={}
for name,(lo,hi) in groups.items():
    sub=pw[(pw['bmi']>=lo)&(pw['bmi']<hi)]
    if len(sub)==0: continue
    ts=sub['tstar']
    reach[name]={str(w):round(float(np.mean(ts<=w)),4) for w in range(10,29)}
out['reach_curve']=reach
pw[['bmi','tstar']].round(3).to_csv('results/pw_tstar.csv',index=False,encoding='utf-8')
with open('results/q2b.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,default=str)
print('Q2B_DONE alpha0.90 result example:',res95)

