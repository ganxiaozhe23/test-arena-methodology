# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, json, sys
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.inspection import permutation_importance
import warnings; warnings.filterwarnings('ignore')
out={}
df=pd.read_csv('results/female_tidy.csv',encoding='utf-8')
# label: aneuploidy present (any of T13/T18/T21/combos) vs normal (N)
df['abnormal']=(df['aneu']!='N').astype(int)
# features per problem4: X concentration, chromosomal Z values, GC content, reads/proportions, BMI
feat=['xc','z13','z18','z21','zx','gc','gc13','gc18','gc21','reads','mapprop','dupprop','uniqreads','filtfrac','bmi','week','age']
# clip extreme Z values / use robust
X=df[feat].copy()
# impute any missing with median
for c in feat:
    X[c]=X[c].fillna(X[c].median())
y=df['abnormal'].values
out['n']=int(len(df)); out['n_abnormal']=int(y.sum()); out['n_normal']=int((y==0).sum())
out['abnormal_rate']=round(float(y.mean()),4)
# normalize
sc=StandardScaler().fit(X); Xs=sc.transform(X)
models={
 'LogisticRegression':LogisticRegression(max_iter=2000,class_weight='balanced'),
 'DecisionTree':DecisionTreeClassifier(max_depth=5,class_weight='balanced',random_state=42),
 'RandomForest':RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42),
 'SVM-RBF':SVC(kernel='rbf',class_weight='balanced',random_state=42,probability=True),
}
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
res={}
for name,m in models.items():
    acc=cross_val_score(m,Xs,y,cv=cv,scoring='accuracy')
    # auc
    if hasattr(m,'predict_proba'):
        auc=cross_val_score(m,Xs,y,cv=cv,scoring='roc_auc')
    else:
        auc=cross_val_score(m,Xs,y,cv=cv,scoring='roc_auc')
    res[name]={'acc_mean':round(float(acc.mean()),4),'acc_std':round(float(acc.std()),4),
               'auc_mean':round(float(auc.mean()),4),'auc_std':round(float(auc.std()),4)}
out['model_comparison']=res
# best model feature importance + detailed report on full fit
m=RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=42)
m.fit(Xs,y)
pred=m.predict(Xs); prob=m.predict_proba(Xs)[:,1]
acc=accuracy_score(y,pred); p,r,f1,_=precision_recall_fscore_support(y,pred,average='binary')
out['rf_train']={'acc':round(float(acc),4),'precision':round(float(p),4),'recall':round(float(r),4),'f1':round(float(f1),4),
   'auc':round(float(roc_auc_score(y,prob)),4)}
# feature importances (Gini)
imp=sorted(zip(feat,m.feature_importances_),key=lambda x:-x[1])
out['rf_importance']={k:round(float(v),4) for k,v in imp}
# permutation importance (marginal)
perm=permutation_importance(m,Xs,y,n_repeats=20,random_state=42)
out['rf_perm_importance']={k:round(float(v),4) for k,v in sorted(zip(feat,perm.importances_mean),key=lambda x:-x[1])}
# SVM best
sv=SVC(kernel='rbf',class_weight='balanced',random_state=42,probability=True)
sv.fit(Xs,y); spp=sv.predict_proba(Xs)[:,1]; sp=sv.predict(Xs)
out['svm_train']={'acc':round(float(accuracy_score(y,sp)),4),'auc':round(float(roc_auc_score(y,spp)),4)}
# Z-value thresholding base rate (clinical standard): label abnormal if max(|z13|,|z18|,|z21|)>3
zthr=df[['z13','z18','z21']].max(axis=1).abs()
pred_z=(zthr>3).astype(int)
out['z_thr3']={'acc':round(float(accuracy_score(y,pred_z)),4),
   'precision':round(float(precision_recall_fscore_support(y,pred_z,average='binary')[0]),4),
   'recall':round(float(precision_recall_fscore_support(y,pred_z,average='binary')[1]),4),
   'f1':round(float(precision_recall_fscore_support(y,pred_z,average='binary')[2]),4)}
# confusion matrix counts for SVM
from sklearn.metrics import confusion_matrix
out['svm_confusion']={'labels':['normal','abnormal'],'matrix':confusion_matrix(y,sp).tolist()}
# per-chromosome breakdown of anomalies for description
out['aneu_counts']=df['aneu'].value_counts(dropna=False).to_dict()
# summary of features by class (for interpretation table)
grp=df.groupby('abnormal')[feat].mean().round(4)
out['feat_means_by_class']=grp.to_dict('index')
with open('results/q4.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,default=str)
print('Q4_DONE n',len(df),'abn',int(y.sum()))

