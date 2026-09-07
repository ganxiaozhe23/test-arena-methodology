# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8')
import statsmodels.api as sm
from scipy.optimize import curve_fit
import warnings; warnings.filterwarnings('ignore')
out={}
d=pd.read_csv('results/male_tidy.csv',encoding='utf-8')
d=d.dropna(subset=['y','week','bmi'])
d=d[d['y']>0]
print('N used',len(d),'unique women',d['code'].nunique())
# correlations
res={}
for v in ['week','bmi','age','height','weight']:
    sub=d[[ 'y',v]].dropna()
    # pearson and spearman
    pr=sub['y'].corr(sub[v],method='pearson')
    sp=sub['y'].corr(sub[v],method='spearman')
    res[v]={'pearson':round(float(pr),4),'spearman':round(float(sp),4)}
out['corr']=res
# bin means by week
d['wk']=d['week'].round(0)
wbin=d.groupby('wk')['y'].agg(['mean','median','count']).round(4)
out['week_bins']=wbin.to_dict('index')
# bin means by bmi
d['bm']=d['bmi'].round()
bbin=d.groupby('bm')['y'].agg(['mean','median','count']).round(4)
out['bmi_bins']=bbin.to_dict('index')
# ---- MODEL COMPARISON ----
def cluster_robust_pres(res, name):
    params=res.params; pvals=res.pvalues
    return {name: {'params':{k:round(float(v),5) for k,v in params.items()},
                   'pvals':{k:('%.3g'%v) for k,v in pvals.items()},
                   'rsq':round(float(res.rsquared),4),'adj_rsq':round(float(res.rsquared_adj),4)}}
# M1 linear y ~ week + bmi
X=d[['week','bmi']].copy(); X=sm.add_constant(X)
m1=sm.OLS(d['y'],X).fit(cov_type='cluster',cov_kwds={'groups':d['code']})
out['M1_linear']={'rsq':round(float(m1.rsquared),4),'adj':round(float(m1.rsquared_adj),4),
  'params':{k:round(float(v),5) for k,v in m1.params.items()},
  'tvalues':{k:round(float(v),3) for k,v in m1.tvalues.items()},
  'pvals':{k:('%g'%v) for k,v in m1.pvalues.items()},
  'nobs':int(m1.nobs)}
# M2 log(y) ~ week + bmi  (multiplicative/exponential)
Y2=np.log(d['y'])
m2=sm.OLS(Y2,X).fit(cov_type='cluster',cov_kwds={'groups':d['code']})
out['M2_exp']={'rsq':round(float(m2.rsquared),4),'adj':round(float(m2.rsquared_adj),4),
  'params':{k:round(float(v),5) for k,v in m2.params.items()},
  'pvals':{k:('%g'%v) for k,v in m2.pvalues.items()},'nobs':int(m2.nobs)}
# M3 log(y) ~ log(week) + bmi (power in week)
X3=sm.add_constant(pd.DataFrame({'logwk':np.log(d['week']),'bmi':d['bmi']}))
m3=sm.OLS(Y2,X3).fit(cov_type='cluster',cov_kwds={'groups':d['code']})
out['M3_power']={'rsq':round(float(m3.rsquared),4),'adj':round(float(m3.rsquared_adj),4),
  'params':{k:round(float(v),5) for k,v in m3.params.items()},
  'pvals':{k:('%g'%v) for k,v in m3.pvalues.items()},'nobs':int(m3.nobs)}
# M4 nonlinear mixed-effects style: y = a * exp(b*week) / (1 + k*bmi)?? 
# Use a mechanistic dilution model: y = a * week^b * (1 + c*bmi)    fit on raw via curve_fit
def f4(wk, bmi, a,b,c):
    return a*np.power(wk,b)/(1+c*(bmi-30))
try:
    popt,pconv=curve_fit(lambda t,*p: f4(t[0],t[1],*p), [d['week'].values,d['bmi'].values], d['y'].values, p0=[0.01,0.5,0.05], maxfev=20000)
    yhat=f4(d['week'].values,d['bmi'].values,*popt)
    ss=1-float(np.sum((d['y'].values-yhat)**2)/np.sum((d['y'].values-d['y'].mean())**2))
    out['M4_nonlinear']={'params':{'a':round(float(popt[0]),6),'b':round(float(popt[1]),5),'c':round(float(popt[2]),5)},'R2':round(ss,4),'popt_list':[round(float(v),6) for v in popt]}
except Exception as e:
    out['M4_nonlinear']={'error':str(e)}
# Nonlinear mixed-ish: use exp model y = exp(b0+b1*week+b2*bmi) with random intercept per woman via MixedLM
try:
    dy=d.copy()
    md=sm.MixedLM.from_formula("y ~ week + bmi", dy, groups=dy["code"]).fit()
    # predict fixed part R2
    fpred=md.fittedvalues
    out['M5_mixedlm']={'rsq':round(float(1-((dy['y'].values-fpred)**2).sum()/((dy['y'].values-dy['y'].mean()**2).sum())),4),
       'params':{k:round(float(v),4) for k,v in md.fe_params.items()},
       'pvals':{k:('%g'%v) for k,v in md.pvalues.items()}}
except Exception as e:
    out['M5_mixedlm']={'error':str(e)}
with open('results/p1.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,default=str)
print('P1_DONE N',len(d))

