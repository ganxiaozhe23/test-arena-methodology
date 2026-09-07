# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
out={}
d=pd.read_csv('results/male_tidy.csv',encoding='utf-8')
d=d.dropna(subset=['y','week','bmi']); d=d[d['y']>0]
d['logy']=np.log(d['y'])
# Recompute per-woman FE threshold (same as q23) for consistency
gm=d.groupby('code')
d['dm_logy']=d['logy']-gm['logy'].transform('mean')
d['dm_week']=d['week']-gm['week'].transform('mean')
import statsmodels.api as sm
Xw=sm.add_constant(d[['dm_week']]); fe=sm.OLS(d['dm_logy'],Xw).fit()
b_w=float(fe.params['dm_week']); out['within_slope']=round(b_w,5)
rows=[]
for code,sub in gm:
    logy_m=sub['logy'].mean(); week_m=sub['week'].mean()
    a=logy_m-b_w*week_m
    t=(np.log(0.04)-a)/b_w
    rows.append({'code':code,'bmi':sub['bmi'].iloc[0],'age':sub['age'].iloc[0],
        'height':sub['height'].iloc[0],'weight':sub['weight'].iloc[0],'t':t,'ymax':sub['y'].max()})
pw=pd.DataFrame(rows)
# treat "never reach 4%": those whose model t>25 or with plateau -> cap at practical max. Keep t as is, but for cdf we use available
# ---- RISK MINIMIZATION ----
# Detection-time risk function from problem text tiers: risk low/medium/high
def g(t):
    if t<=12: return 1.0      # early: low risk
    elif t<=27: return 3.0    # mid: high risk
    else: return 6.0          # late: very high
def g_smooth(t):
    # continuous smooth surrogate: risk increases with week
    return 1.0*np.exp(0.06*(t-12))
groups={
 'G1_BMI_20_28':(20,28),'G2_BMI_28_32':(28,32),'G3_BMI_32_36':(32,36),
 'G4_BMI_36_40':(36,40),'G5_BMI_40_plus':(40,120)}
# Also derive "optimal group bounds" later. For now use given.
res={}
for name,(lo,hi) in groups.items():
    sub=pw[(pw['bmi']>=lo)&(pw['bmi']<hi)]
    if len(sub)==0:
        res[name]={'n':0}; continue
    ts=np.clip(sub['t'].values,10,28)
    # empirical fraction reaching threshold by week w
    fracreach=lambda w: float(np.mean(ts<=w))
    # minimize expected risk over choice of detection week w
    best=None
    for w in np.arange(10,26.01,0.5):
        A=fracreach(w)   # P(>=4% by week w) = 达标比例
        # if not yet reliable (concentration<4%), the detection "misses"/fails -> falls back to later
        # expected risk = A*g_smooth(w) + (1-A)*g_smooth(w + 2)  (retest 2 weeks later)
        E=A*g_smooth(w)+(1-A)*g_smooth(w+2)
        if best is None or E<best[1]:
            best=(w,E,A)
    res[name]={'n':int(len(sub)),'bmi_mean':round(float(sub['bmi'].mean()),2),
      'bmi_range':[round(float(sub['bmi'].min()),2),round(float(sub['bmi'].max()),2)],
      'best_week':round(float(best[0]),1),'expected_risk':round(float(best[1]),4),
      'reach_prob':round(float(best[2]),4)}
out['risk_min_Q2']=res
# table of reach-probability at candidate weeks per group (for figure)
reach={}
for name,(lo,hi) in groups.items():
    sub=pw[(pw['bmi']>=lo)&(pw['bmi']<hi)]
    if len(sub)==0: continue
    ts=np.clip(sub['t'].values,10,28)
    reach[name]={str(int(w)):round(float(np.mean(ts<=w)),4) for w in range(10,26)}
out['reach_by_week']=reach
# 5-group optimal vs risk sensitivity to smooth param
sens={}
for k_param in [0.04,0.06,0.08]:
    def gs(t): return 1.0*np.exp(k_param*(t-12))
    r={}
    for name,(lo,hi) in groups.items():
        sub=pw[(pw['bmi']>=lo)&(pw['bmi']<hi)]
        if len(sub)==0: continue
        ts=np.clip(sub['t'].values,10,28)
        best=None
        for w in np.arange(10,26.01,0.5):
            A=float(np.mean(ts<=w)); E=A*gs(w)+(1-A)*gs(w+2)
            if best is None or E<best[1]: best=(w,E,A)
        r[name]=round(float(best[0]),1)
    sens[str(k_param)]=r
out['sens_week_vs_k']=sens
with open('results/q2.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,default=str)
print('Q2_DONE')

