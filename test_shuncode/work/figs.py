# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, json, sys, os
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
os.makedirs('figures',exist_ok=True)
plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei','Arial Unicode MS']
plt.rcParams['axes.unicode_minus']=False
d=pd.read_csv('results/male_tidy.csv',encoding='utf-8'); d=d.dropna(subset=['y','week','bmi']); d=d[d['y']>0]
# FIG1 scatter y vs week colored by bmi, plus y vs bmi
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
sc=ax[0].scatter(d['week'],d['y'],c=d['bmi'],cmap='viridis',s=14,alpha=0.6)
ax[0].axhline(0.04,color='red',ls='--',lw=1,label='4% 阈值')
ax[0].set_xlabel('孕周 (周)'); ax[0].set_ylabel('Y染色体浓度'); ax[0].set_title('(a) Y浓度 与 孕周')
cb=fig.colorbar(sc,ax=ax[0]); cb.set_label('BMI')
ax[1].scatter(d['bmi'],d['y'],c=d['week'],cmap='plasma',s=14,alpha=0.6)
ax[1].axhline(0.04,color='red',ls='--',lw=1,label='4% 阈值')
ax[1].set_xlabel('孕前 BMI (kg/m²)'); ax[1].set_ylabel('Y染色体浓度'); ax[1].set_title('(b) Y浓度 与 BMI')
plt.tight_layout(); plt.savefig('figures/fig1_relation.png',dpi=140); plt.close()
# FIG2 reach-curve for 5 BMI groups (from q2b reach_curve)
q=json.load(open('results/q2b.json'))
rc=q['reach_curve']
fig,ax=plt.subplots(figsize=(8,4.6))
for name,cv in rc.items():
    wk=sorted(int(k) for k in cv); vals=[cv[str(w)] for w in wk]
    ax.plot(wk,vals,marker='o',ms=3,label=name.replace('G',''),lw=1.8)
ax.axhline(0.90,color='gray',ls=':',lw=1); ax.text(11.2,0.905,'90% 达标线',color='gray',fontsize=8)
ax.axhline(0.95,color='gray',ls=':',lw=1); ax.text(11.2,0.955,'95% 达标线',color='gray',fontsize=8)
ax.set_xlabel('NIPT 检测孕周'); ax.set_ylabel('达标比例 (P(Y≥4%))'); ax.set_title('各组 达标比例-孕周 曲线 (问题二)')
ax.legend(title='BMI组'); ax.grid(alpha=0.3); ax.set_ylim(0,1.05)
plt.tight_layout(); plt.savefig('figures/fig2_reach.png',dpi=140); plt.close()
# FIG3 KMeans + optimal week (Q3) reach curves by label
q3=json.load(open('results/q3.json'))
fig,ax=plt.subplots(figsize=(8,4.6))
for lb,cv in q3['Q3_reach_bylabel'].items():
    wk=sorted(float(k) for k in cv); vals=[cv[str(k)] for k in wk]
    ax.plot(wk,vals,marker='o',ms=3,lw=1.8,label=f"组{lb} (BMI均值)"
            )
ax.axhline(0.90,color='gray',ls=':',lw=1)
ax.set_xlabel('NIPT 检测孕周'); ax.set_ylabel('达标比例 (含检测误差)'); ax.set_title('多因素+误差 达标曲线与最优时点 (问题三)')
ax.legend(title='K-means组'); ax.grid(alpha=0.3); ax.set_ylim(0,1.05)
plt.tight_layout(); plt.savefig('figures/fig3_reach_mc.png',dpi=140); plt.close()
# FIG4 female model comparison + feature importance
q4=json.load(open('results/q4.json'))
labels=list(q4['model_comparison'].keys()); acc=[q4['model_comparison'][k]['acc_mean'] for k in labels]
auc=[q4['model_comparison'][k]['auc_mean'] for k in labels]
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
x=np.arange(len(labels)); w=0.35
ax[0].bar(x-w/2,acc,w,label='准确率',color='#4C72B0'); ax[0].bar(x+w/2,auc,w,label='AUC',color='#DD8452')
ax[0].set_xticks(x); ax[0].set_xticklabels(labels,rotation=15); ax[0].set_ylim(0.5,1.0)
ax[0].set_ylabel('得分'); ax[0].set_title('(a) 分类器比较 (5折交叉验证)'); ax[0].legend()
imp=q4['rf_importance']; keys=list(imp)[:8]; vals=[imp[k] for k in keys]
ax[1].barh(keys[::-1],vals[::-1],color='#55A868'); ax[1].set_xlabel('Gini 重要性'); ax[1].set_title('(b) 随机森林特征重要性 (前8)')
plt.tight_layout(); plt.savefig('figures/fig4_classifier.png',dpi=140); plt.close()
# FIG5 within-woman tstar vs BMI scatter
pw=pd.read_csv('results/pw_tstar.csv',encoding='utf-8')
fig,ax=plt.subplots(figsize=(7,4.4))
ax.scatter(pw['bmi'],pw['tstar'],s=22,alpha=0.6,c='#4C72B0')
ax.set_xlabel('孕前 BMI (kg/m²)'); ax.set_ylabel('最早达标孕周 (Y≥4%)'); ax.set_title('最早达标孕周 与 BMI 的关系')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('figures/fig5_tstar_bmi.png',dpi=140); plt.close()
print('FIGURES_DONE',os.listdir('figures'))

