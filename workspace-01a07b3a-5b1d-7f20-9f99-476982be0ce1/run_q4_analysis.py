import pandas as pd
import numpy as np
import scipy.stats as stats
import json
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

# 1. Load female data
female = pd.read_csv('test-arena-methodology/女胎检测数据.csv')
female.columns = [c.strip() for c in female.columns]

# Impute missing BMI
if female['孕妇BMI'].isna().any():
    female['孕妇BMI'] = female['孕妇BMI'].fillna(female['体重'] / ((female['身高']/100)**2))

def parse_ga(val):
    if pd.isna(val): return np.nan
    val = str(val).strip()
    import re
    m = re.match(r'^(\d+)[wW](?:\+(\d+))?$', val)
    if m:
        w = int(m.group(1))
        d = int(m.group(2)) if m.group(2) else 0
        return w + d / 7.0
    return np.nan

female['GA'] = female['检测孕周'].apply(parse_ga)

# Targets
female['is_abnormal'] = female['染色体的非整倍体'].notna().astype(int)
female['is_T13'] = female['染色体的非整倍体'].str.contains('T13', na=False).astype(int)
female['is_T18'] = female['染色体的非整倍体'].str.contains('T18', na=False).astype(int)
female['is_T21'] = female['染色体的非整倍体'].str.contains('T21', na=False).astype(int)

# Feature engineering
female['GC_diff_13'] = female['13号染色体的GC含量'] - female['GC含量']
female['GC_diff_18'] = female['18号染色体的GC含量'] - female['GC含量']
female['GC_diff_21'] = female['21号染色体的GC含量'] - female['GC含量']
female['unique_ratio'] = female['唯一比对的读段数'] / female['原始读段数']
female['dup_ratio'] = female['重复读段的比例']
female['filter_ratio'] = female['被过滤掉读段数的比例']

feat_cols = [
    '年龄', '身高', '体重', '孕妇BMI', 'GA',
    '原始读段数', '在参考基因组上比对的比例', '重复读段的比例', '唯一比对的读段数', 'GC含量',
    '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值', 'X染色体浓度',
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', '被过滤掉读段数的比例',
    'GC_diff_13', 'GC_diff_18', 'GC_diff_21', 'unique_ratio'
]

X = female[feat_cols].values
y = female['is_abnormal'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
}

model_results = {}
oof_predictions = {}

for name, model in models.items():
    oof_prob = np.zeros(len(y))
    oof_pred = np.zeros(len(y))
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = (X_scaled[train_idx], X_scaled[val_idx]) if 'Logistic' in name else (X[train_idx], X[val_idx])
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        oof_prob[val_idx] = probs
        oof_pred[val_idx] = (probs >= 0.5).astype(int)
        
    auc = roc_auc_score(y, oof_prob)
    acc = accuracy_score(y, oof_pred)
    prec = precision_score(y, oof_pred, zero_division=0)
    rec = recall_score(y, oof_pred)
    f1 = f1_score(y, oof_pred)
    brier = brier_score_loss(y, oof_prob)
    cm = confusion_matrix(y, oof_pred).tolist()
    
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y, oof_pred).ravel()
    spec = tn / (tn + fp)
    
    model_results[name] = {
        'ROC_AUC': float(auc),
        'Accuracy': float(acc),
        'Sensitivity_Recall': float(rec),
        'Specificity': float(spec),
        'Precision': float(prec),
        'F1_Score': float(f1),
        'Brier_Score': float(brier),
        'Confusion_Matrix': cm
    }
    oof_predictions[name] = oof_prob.tolist()

# Feature importance from Random Forest
rf_full = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)
rf_full.fit(X, y)
feat_imp = pd.DataFrame({
    'feature': feat_cols,
    'importance': rf_full.feature_importances_
}).sort_values('importance', ascending=False)
feat_imp.to_csv('analysis_output/q4_feature_importance.csv', index=False)

# Specific trisomy models (T13, T18, T21)
trisomies = ['T13', 'T18', 'T21']
tri_results = {}
for tri in trisomies:
    y_tri = female[f'is_{tri}'].values
    rf_tri = RandomForestClassifier(n_estimators=150, max_depth=5, class_weight='balanced', random_state=42)
    oof_tri = np.zeros(len(y_tri))
    for train_idx, val_idx in cv.split(X, y_tri):
        rf_tri.fit(X[train_idx], y_tri[train_idx])
        oof_tri[val_idx] = rf_tri.predict_proba(X[val_idx])[:, 1]
    tri_auc = roc_auc_score(y_tri, oof_tri) if len(np.unique(y_tri)) > 1 else 0.5
    tri_results[tri] = {
        'positives': int(y_tri.sum()),
        'ROC_AUC': float(tri_auc)
    }

# Clinical Rule-based Model (GC-corrected Z-score cutoff)
# Regression of Z-scores on GC content on normal cohort
norm_mask = female['is_abnormal'] == 0
z13_reg = stats.linregress(female.loc[norm_mask, '13号染色体的GC含量'], female.loc[norm_mask, '13号染色体的Z值'])
z18_reg = stats.linregress(female.loc[norm_mask, '18号染色体的GC含量'], female.loc[norm_mask, '18号染色体的Z值'])
z21_reg = stats.linregress(female.loc[norm_mask, '21号染色体的GC含量'], female.loc[norm_mask, '21号染色体的Z值'])

z_adj_13 = female['13号染色体的Z值'] - (z13_reg.intercept + z13_reg.slope * female['13号染色体的GC含量'])
z_adj_18 = female['18号染色体的Z值'] - (z18_reg.intercept + z18_reg.slope * female['18号染色体的GC含量'])
z_adj_21 = female['21号染色体的Z值'] - (z21_reg.intercept + z21_reg.slope * female['21号染色体的GC含量'])

# Combined clinical score = max(|Z_adj|) combined with X concentration penalty
# When X conc is significantly negative or Z_adj is extreme
clinical_score = np.maximum.reduce([np.abs(z_adj_13), np.abs(z_adj_18), np.abs(z_adj_21)]) - 20.0 * np.minimum(0, female['X染色体浓度'])
clin_auc = roc_auc_score(y, clinical_score)
model_results['GC-Corrected Clinical Z-Score Rule'] = {
    'ROC_AUC': float(clin_auc),
    'Description': 'Multi-chromosome GC-corrected adaptive Z-score + X-concentration penalty'
}

q4_output = {
    'dataset_info': {
        'total_samples': len(female),
        'abnormal_count': int(y.sum()),
        'normal_count': int((1-y).sum()),
        'prevalence': float(y.mean())
    },
    'model_evaluations': model_results,
    'trisomy_specific_models': tri_results,
    'top_features': feat_imp.head(10).to_dict(orient='records')
}

with open('analysis_output/q4_output.json', 'w') as f:
    json.dump(q4_output, f, indent=2)

print('Q4 analysis completed successfully!')
