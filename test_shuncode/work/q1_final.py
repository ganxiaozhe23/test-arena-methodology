# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8')
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings; warnings.filterwarnings('ignore')
out={}
d=pd.read_csv('results/male_tidy.csv',encoding='utf-8')
d=d.dropna(subset=['y','week','bmi']); d=d[d['y']>0]
d['logy']=np.log(d['y'])
# ---- FINAL MODEL: log y = b0 + b1*log(week) + b2*bmi + b3*age + b4*height + b5*weight (random intercept per woman) ----
# Because y ~ week^b1 * exp(b2*bmi ...) multiplicative => log linear. Fit MixedLM.
d['logwk']=np.log(d['week'])
# base model (week+bmi), full model (add age,height,weight)
form_base="logy ~ logwk + bmi"
form_full="logy ~ logwk + bmi + age + height + weight"
mb=MixedLM.from_formula(form_base, d, groups=d['code']).fit()
mf=MixedLM.from_formula(form_full, d, groups=d['code']).fit()
def summ(m):
    return {'params':{k:round(float(v),5) for k,v in m.fe_params.items()},
            'bse':{k:round(float(v),5) for k,v in m.bse_fe.items()},
            'pvals':{k:('%g'%v) for k,v in m.pvalues.items()},
            'loglike':round(float(m.llf),4),
            'group_sd':round(float(m.cov_re.iloc[0,0]**0.5),6),
            'scale_sd':round(float(m.scale**0.5),6)}
out['base']=summ(mb)
out['full']=summ(mf)
# Within-woman vs between-woman: R2 for fixed-effects-only prediction
import numpy.linalg as la
def fixed_r2(m):
    d2=d.copy()
    d2['pred']=m.fittedvalues
    # random effect part not included in fixed; use predict on fixed only via model
    return None
out['base_formula']="ln(y)=b0 + b1*ln(week) + b2*bmi + u_i + eps"
out['full_formula']="ln(y)=b0 + b1*ln(week) + b2*bmi + b3*age + b4*height + b5*weight + u_i + eps"
# ---- Overall R2 of fixed part (marginal) using MixedLM mean structure ----
# Compute pseudo R2 using fixed-effects OLS (conventional) for reference
X=sm.add_constant(d[['logwk','bmi']])
ols=sm.OLS(d['logy'],X).fit()
out['ols_base_log']={'rsq':round(float(ols.rsquared),4),'adj':round(float(ols.rsquared_adj),4),
   'params':{k:round(float(v),5) for k,v in ols.params.items()},
   'tvals':{k:round(float(v),3) for k,v in ols.tvalues.items()},
   'pvals':{k:('%g'%v) for k,v in ols.pvalues.items()},
   'f':round(float(ols.fvalue),3),'f_pval':('%g'%ols.f_pvalue),
   'aic':round(float(ols.aic),2),'bic':round(float(ols.bic),2)}
# ---- Likelihood-ratio test BMI significance within mixed model ----
try:
    m_no_bmi=MixedLM.from_formula("logy ~ logwk", d, groups=d['code']).fit()
    lr=2*(mb.llf-m_no_bmi.llf)
    p=1-__import__('scipy').stats.chi2.cdf(lr,1)
    out['lr_bmi']={'LR':round(float(lr),3),'df':1,'pval':('%g'%p)}
except Exception as e:
    out['lr_bmi']={'error':str(e)}
# ---- derived threshold week function: for a woman with given BMI, type-II (fixed part) & solve y=0.04 ----
# From base: log y = b0 + b1*log wk + b2*bmi (+u). Set y=0.04 -> logwk = (log(0.04)-b0-b2*bmi)/b1
b0=mb.fe_params['Intercept']; b1=mb.fe_params['logwk']; b2=mb.fe_params['bmi']
def thr_week(bmi):
    logwk=(np.log(0.04)-b0-b2*bmi)/b1
    return float(np.exp(logwk))
out['thr_coeffs']={'b0':round(float(b0),5),'b1':round(float(b1),5),'b2':round(float(b2),5)}
# threshold week for a grid of BMI (fixed-effects, no random effect = population average)
grid=[20,22,24,26,28,30,32,34,36,38,40,42,44,46]
out['thr_week_population']={str(k):round(thr_week(k),3) for k in grid}
# Correlation matrix of numeric predictors
preds=['week','bmi','age','height','weight']
corr=d[preds+['y']].corr(method='spearman')
out['corr_matrix_spearman']={r:{c:round(float(corr.loc[r,c]),3) for c in corr.columns} for r in corr.index}
with open('results/q1_final.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,default=str)
print('Q1FINAL_DONE')

