# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, re, sys
sys.stdout.reconfigure(encoding='utf-8')
def parse_wk(s):
    s=str(s).strip()
    m=re.match(r'(\d+)w\+(\d+)',s)
    if m: return int(m.group(1))+int(m.group(2))/7.0
    m=re.match(r'(\d+)w',s)
    if m: return float(m.group(1))
    return np.nan
def enc_aneu(s):
    if pd.isna(s): return 'N'
    return str(s)
def enc_health(s):
    if s=='是': return 1
    if s=='否': return 0
    return np.nan
def enc_ivf(s):
    if str(s).strip()=='自然受孕': return 0
    if 'IVF' in str(s): return 2
    if 'IUI' in str(s): return 1
    return -1
def enc_preg(s):
    if str(s).strip()=='1': return 1
    if str(s).strip()=='2': return 2
    return 3
male=pd.read_csv('男胎检测数据.csv',encoding='utf-8-sig')
fem=pd.read_csv('女胎检测数据.csv',encoding='utf-8-sig')
# male export
mm=male[['序号','孕妇代码','年龄','身高','体重','检测孕周','孕妇BMI','Y染色体浓度','Y染色体的Z值','X染色体浓度','GC含量','13号染色体的GC含量','18号染色体的GC含量','21号染色体的GC含量','原始读段数','在参考基因组上比对的比例','重复读段的比例','唯一比对的读段数  ','被过滤掉读段数的比例','检测抽血次数','染色体的非整倍体','胎儿是否健康','怀孕次数','IVF妊娠','生产次数']].copy()
mm['week']=mm['检测孕周'].apply(parse_wk)
mm['aneu']=mm['染色体的非整倍体'].apply(enc_aneu)
mm['health']=mm['胎儿是否健康'].apply(enc_health)
mm['ivf']=mm['IVF妊娠'].apply(enc_ivf)
mm['preg']=mm['怀孕次数'].apply(enc_preg)
mm=mm.rename(columns={'序号':'sample','孕妇代码':'code','年龄':'age','身高':'height','体重':'weight','孕妇BMI':'bmi','Y染色体浓度':'y','Y染色体的Z值':'yz','X染色体浓度':'xc','GC含量':'gc','13号染色体的GC含量':'gc13','18号染色体的GC含量':'gc18','21号染色体的GC含量':'gc21','原始读段数':'reads','在参考基因组上比对的比例':'mapprop','重复读段的比例':'dupprop','唯一比对的读段数  ':'uniqreads','被过滤掉读段数的比例':'filtfrac','检测抽血次数':'bleed','生产次数':'parity'})
mcols=['sample','code','age','height','weight','week','bmi','y','yz','xc','gc','gc13','gc18','gc21','reads','mapprop','dupprop','uniqreads','filtfrac','bleed','aneu','health','ivf','preg','parity']
mm=mm[mcols]
mm.to_csv('results/male_tidy.csv',index=False,encoding='utf-8')
# female export
ff=fem[['序号','孕妇代码','年龄','身高','体重','检测孕周','孕妇BMI','X染色体浓度','13号染色体的Z值','18号染色体的Z值','21号染色体的Z值','X染色体的Z值','GC含量','13号染色体的GC含量','18号染色体的GC含量','21号染色体的GC含量','原始读段数','在参考基因组上比对的比例','重复读段的比例','唯一比对的读段数','被过滤掉读段数的比例','检测抽血次数','染色体的非整倍体','怀孕次数','IVF妊娠','生产次数']].copy()
ff['week']=ff['检测孕周'].apply(parse_wk)
ff['aneu']=ff['染色体的非整倍体'].apply(enc_aneu)
ff['ivf']=ff['IVF妊娠'].apply(enc_ivf)
ff['preg']=ff['怀孕次数'].apply(enc_preg)
ff=ff.rename(columns={'序号':'sample','孕妇代码':'code','年龄':'age','身高':'height','体重':'weight','孕妇BMI':'bmi','X染色体浓度':'xc','13号染色体的Z值':'z13','18号染色体的Z值':'z18','21号染色体的Z值':'z21','X染色体的Z值':'zx','GC含量':'gc','13号染色体的GC含量':'gc13','18号染色体的GC含量':'gc18','21号染色体的GC含量':'gc21','原始读段数':'reads','在参考基因组上比对的比例':'mapprop','重复读段的比例':'dupprop','唯一比对的读段数':'uniqreads','被过滤掉读段数的比例':'filtfrac','检测抽血次数':'bleed','生产次数':'parity'})
fcols=['sample','code','age','height','weight','week','bmi','xc','z13','z18','z21','zx','gc','gc13','gc18','gc21','reads','mapprop','dupprop','uniqreads','filtfrac','bleed','aneu','ivf','preg','parity']
ff=ff[fcols]
ff.to_csv('results/female_tidy.csv',index=False,encoding='utf-8')
print('male rows',len(mm),'female rows',len(ff))
print('male week invalid',int(mm['week'].isna().sum()),'female week invalid',int(ff['week'].isna().sum()))
print('male y range',round(mm['y'].min(),4),round(mm['y'].max(),4))

