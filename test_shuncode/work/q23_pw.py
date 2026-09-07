# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8')
import statsmodels.api as sm
import warnings; warnings.filterwarnings('ignore')
out={}
d=pd.read_csv('results/male_tidy.csv',encoding='utf-8')
d=d.dropna(subset=['y','week','bmi']); d=d[d['y']>0]
d['logy']=np.log(d['y'])
# ---- WITHIN-WOMAN FE (demeaned) estimator for week slope ----
gm=d.groupby('code')
d['dm_logy']=d['logy']-gm['logy'].transform('mean')
d['dm_week']=d['week']-gm['week'].transform('mean')
Xw=d[['dm_week']]; Xw=sm.add_constant(Xw)
fe=sm.OLS(d['dm_logy'],Xw).fit()
b_w=fe.params['dm_week']
out['within_slope_week']=round(float(b_w),6)
out['within_pval']=('%g'%fe.pvalues['dm_week'])
# woman-specific intercept a_i: mean logy - b_w * mean week
summary=[]
for code,sub in gm:
    logy_m=sub['logy'].mean(); week_m=sub['week'].mean()
    a=logy_m - b_w*week_m
    t_star=(np.log(0.04)-a)/b_w
    summary.append({'code':code,'bmi':sub['bmi'].iloc[0],'age':sub['age'].iloc[0],
        'height':sub['height'].iloc[0],'weight':sub['weight'].iloc[0],
        'n':len(sub),'t_star':t_star,'a':a,'ymax':sub['y'].max()})
pw=pd.DataFrame(summary)
pw['t_star']=pw['t_star'].clip(10,30)
out['tstar_desc']=pw['t_star'].describe().round(3).to_dict()
# relation t_star ~ bmi (and factors) via OLS
X=sm.add_constant(pw[['bmi']])
lr=sm.OLS(pw['t_star'],X).fit()
out['tstar_bmi_ols']={'params':{k:round(float(v),4) for k,v in lr.params.items()},
   'pvals':{k:('%g'%v) for k,v in lr.pvalues.items()},'rsq':round(float(lr.rsquared),4),
   'adj':round(float(lr.rsquared_adj),4)}
# full regression
Xf=sm.add_constant(pw[['bmi','age','height','weight']])
lrf=sm.OLS(pw['t_star'],Xf).fit()
out['tstar_full_ols']={'params':{k:round(float(v),4) for k,v in lrf.params.items()},
   'pvals':{k:('%g'%v) for k,v in lrf.pvalues.items()},'rsq':round(float(lrf.rsquared),4),
   'adj':round(float(lrf.rsquared_adj),4),'f':round(float(lrf.fvalue),3),'f_p':('%g'%lrf.f_pvalue)}
# by BMI bucket summary of t_star
pw['bmi_grp']=pd.cut(pw['bmi'],bins=[20,28,32,36,40,100],labels=['[20,28)','[28,32)','[32,36)','[36,40)','>=40'])
tab=pw.groupby('bmi_grp')['t_star'].agg(['count','mean','median','std']).round(3)
out['tstar_by_bmi']=tab.to_dict('index')
# 4% at given week (reliability prob) via within-model: only woman-specific, so group-level probability:
# For groups we compute fraction reaching threshold by a given week from the t_star distribution.
for grp in ['[20,28)','[28,32)','[32,36)','[36,40)','>=40']:
    sub=pw[pw['bmi_grp']==grp]['t_star']
    cdf={}
    for w in [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]:
        cdf[str(w)]=round(float((sub<=w).mean()),4)
    out['frac_reach_by_week_'+grp]=cdf
with open('results/q23_pw.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,default=str)
print('Q23_DONE within_slope',round(float(b_w),5),'nsubset',len(pw))

