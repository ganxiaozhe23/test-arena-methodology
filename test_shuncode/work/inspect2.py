# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8')
out={}
def wk_vals(df):
    return [str(x) for x in df['检测孕周'].astype(str).unique()[:40]]
def cat(df, col, n_limit=30):
    vc=df[col].value_counts(dropna=False)
    return {str(k):int(v) for k,v in vc.items()}
male=pd.read_csv('男胎检测数据.csv',encoding='utf-8-sig')
fem=pd.read_csv('女胎检测数据.csv',encoding='utf-8-sig')
out['male_weeks_sample']=wk_vals(male)
out['male_weeks_nunique']=male['检测孕周'].nunique()
out['male_aneuploidy']=cat(male,'染色体的非整倍体')
out['male_health']=cat(male,'胎儿是否健康')
out['male_ivf']=cat(male,'IVF妊娠')
out['male_preg']=cat(male,'怀孕次数')
out['male_bleed']=cat(male,'检测抽血次数')
out['male_code_n']=male['孕妇代码'].nunique()
out['male_dupe_groups']=int(male.groupby('孕妇代码').size().median())
out['fem_weeks_sample']=wk_vals(fem)
out['fem_weeks_nunique']=fem['检测孕周'].nunique()
out['fem_aneuploidy']=cat(fem,'染色体的非整倍体')
out['fem_health']=cat(fem,'胎儿是否健康')
out['fem_ivf']=cat(fem,'IVF妊娠')
out['fem_preg']=cat(fem,'怀孕次数')
out['fem_bleed']=cat(fem,'检测抽血次数')
out['fem_code_n']=fem['孕妇代码'].nunique()
# rows where aneuploidy non-null in male, show z-values
man=male[male['染色体的非整倍体'].notna()]
cols=['孕妇代码','检测孕周','孕妇BMI','21号染色体的Z值','18号染色体的Z值','13号染色体的Z值','染色体的非整倍体','胎儿是否健康']
out['male_abn']=man[cols].to_dict('records')
for r in out['male_abn']:
    for k in list(r.keys()):
        if k!='孕妇代码':
            if isinstance(r[k],float): r[k]=round(r[k],3)
        if pd.isna(r[k]): r[k]=None
fen=fem[fem['染色体的非整倍体'].notna()]
fcols=['孕妇代码','检测孕周','孕妇BMI','21号染色体的Z值','18号染色体的Z值','13号染色体的Z值','X染色体的Z值','染色体的非整倍体','胎儿是否健康']
out['fem_abn']=fen[fcols].to_dict('records')
for r in out['fem_abn']:
    for k in list(r.keys()):
        if k!='孕妇代码':
            if isinstance(r[k],float): r[k]=round(r[k],3)
        if pd.isna(r[k]): r[k]=None
with open('results/inspect2.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,default=str)
print('INSPECT2_WRITTEN')

