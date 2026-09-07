# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8')
out={}
d=pd.read_csv('results/male_tidy.csv',encoding='utf-8')
d=d.dropna(subset=['y','week']); d=d[d['y']>0]
g=d.sort_values('week').groupby('code')
# per woman: min week, max week, n, week span, min y, max y, whether ever >=0.04
rows=[]
for code,sub in g:
    sub=sub[['week','y','bmi','age','height','weight']].sort_values('week')
    rows.append({'code':code,'n':len(sub),'wmin':sub['week'].min(),'wmax':sub['week'].max(),
                 'ymin':sub['y'].min(),'ymax':sub['y'].max(),'bmi':sub['bmi'].iloc[0],
                 'age':sub['age'].iloc[0],'height':sub['height'].iloc[0],'weight':sub['weight'].iloc[0],
                 'ever_ge4':int((sub['y']>=0.04).any()),'first_ge4_week':(sub.loc[sub['y']>=0.04,'week'].min() if (sub['y']>=0.04).any() else np.nan)})
pw=pd.DataFrame(rows)
out['n_women']=len(pw)
out['obs_per_women_desc']=pw['n'].describe().round(2).to_dict()
out['ever_ge4_rate']=round(pw['ever_ge4'].mean(),3)
out['ymin_desc']=pw['ymin'].describe().round(4).to_dict()
out['ymax_desc']=pw['ymax'].describe().round(4).to_dict()
# women who never reach 4%
never=pw[pw['ever_ge4']==0]
out['n_never_ge4']=len(never)
out['never_ge4_ymax_desc']=never['ymax'].describe().round(4).to_dict()
# first_ge4_week distribution by bmi bucket
pw['bmi_grp']=pd.cut(pw['bmi'],bins=[20,28,32,36,40,100],labels=['[20,28)','[28,32)','[32,36)','[36,40)','>=40'])
tab=pw.groupby('bmi_grp').agg(n=('code','count'),median_first=('first_ge4_week','median'),
    mean_first=('first_ge4_week','mean'),med_ymax=('ymax','median'),rate_ge4=('ever_ge4','mean'))
out['bmi_grp_first_ge4']=tab.round(3).to_dict('index')
# scatter ready data (for figure) sampled
out['pw_sample']=pw[['bmi','first_ge4_week','ymax','n']].round(4).to_dict('records')
with open('results/pw.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,default=str)
print('PW_DONE nwomen',len(pw),'ever_ge4',round(pw['ever_ge4'].mean(),3))

