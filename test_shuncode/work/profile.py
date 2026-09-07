# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, json, io, sys
sys.stdout.reconfigure(encoding='utf-8')
out={}
def prof(df):
    d={}
    d['shape']=[int(df.shape[0]),int(df.shape[1])]
    cols=[str(c) for c in df.columns]
    d['columns']=cols
    d['dtypes']={str(c):str(t) for c,t in df.dtypes.items()}
    d['null_counts']={str(c):int(df[c].isna().sum()) for c in df.columns}
    d['n_unique']={str(c):int(df[c].nunique()) for c in df.columns}
    # numeric cols summary
    num=df.select_dtypes(include=[np.number]).columns
    summ={}
    for c in num:
        s=df[c].dropna()
        summ[str(c)]={'min':round(float(s.min()),4),'max':round(float(s.max()),4),'mean':round(float(s.mean()),4),'median':round(float(s.median()),4),'std':round(float(s.std()),4)}
    d['numeric_summary']=summ
    return d
out['male']=prof(pd.read_csv('男胎检测数据.csv',encoding='utf-8-sig'))
out['female']=prof(pd.read_csv('女胎检测数据.csv',encoding='utf-8-sig'))
with open('results/profile.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,default=str)
print('PROFILE_WRITTEN')

