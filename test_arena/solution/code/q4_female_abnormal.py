import pandas as pd, numpy as np, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
warnings = __import__('warnings'); warnings.filterwarnings('ignore')

df_f = pd.read_csv("/home/user/test_arena/女胎检测数据.csv", encoding='utf-8-sig')
def parse_gw(s):
    m=re.match(r"(\d+)w(?:\+(\d+))?", str(s).strip())
    if m:
        w=int(m.group(1)); d=int(m.group(2)) if m.group(2) else 0
        return w+d/7
    return np.nan
df_f["GW"] = df_f["检测孕周"].apply(parse_gw)
df_f["BMI"] = df_f["孕妇BMI"]
df_f["age"] = df_f["年龄"]
df_f["height"] = df_f["身高"]
df_f["weight"] = df_f["体重"]
# abnormal definition: AB column non-empty => any T13/T18/T21
df_f["abnormal"] = df_f["染色体的非整倍体"].notna() & (df_f["染色体的非整倍体"].astype(str).str.strip() != "")
# Map to binary
df_f["abnormal_bin"] = df_f["abnormal"].astype(int)
print("Female shape:", df_f.shape)
print("Abnormal distribution:")
print(df_f["染色体的非整倍体"].value_counts(dropna=False).head(20))
print("Binary:", df_f["abnormal_bin"].value_counts())
print(f"Abnormal rate: {df_f['abnormal_bin'].mean():.3f}")

# Features: X Z, 13/18/21 Z, GC content, read counts, proportions, BMI, etc.
# List cols
cols = df_f.columns.tolist()
print(cols)
# Check Z values distribution
z_cols = ["13号染色体的Z值","18号染色体的Z值","21号染色体的Z值","X染色体的Z值"]
gc_cols = ["13号染色体的GC含量","18号染色体的GC含量","21号染色体的GC含量","GC含量"]
other = ["原始读段数","在参考基因组上比对的比例","重复读段的比例","唯一比对的读段数","被过滤掉读段数的比例","X染色体浓度","孕妇BMI","年龄","身高","体重","GW"]

# EDA: compare abnormal vs normal Z distributions
for z in z_cols:
    print(f"\n{z}: normal mean {df_f[df_f['abnormal_bin']==0][z].mean():.2f} sd {df_f[df_f['abnormal_bin']==0][z].std():.2f} | abnormal mean {df_f[df_f['abnormal_bin']==1][z].mean():.2f} sd {df_f[df_f['abnormal_bin']==1][z].std():.2f}")
    from scipy.stats import ttest_ind, mannwhitneyu
    try:
        t, p = ttest_ind(df_f[df_f['abnormal_bin']==0][z].dropna(), df_f[df_f['abnormal_bin']==1][z].dropna(), equal_var=False)
        print(f"  t-test p={p:.2e}")
    except: pass
    try:
        u,p2 = mannwhitneyu(df_f[df_f['abnormal_bin']==0][z].dropna(), df_f[df_f['abnormal_bin']==1][z].dropna())
        print(f"  mann p={p2:.2e}")
    except: pass

# Also for GC etc.
for g in gc_cols:
    print(f"{g}: normal {df_f[df_f['abnormal_bin']==0][g].mean():.4f} abnormal {df_f[df_f['abnormal_bin']==1][g].mean():.4f}")

# Build classification dataset dropping rows with missing in key features
feature_list = ["13号染色体的Z值","18号染色体的Z值","21号染色体的Z值","X染色体的Z值","13号染色体的GC含量","18号染色体的GC含量","21号染色体的GC含量","GC含量","原始读段数","在参考基因组上比对的比例","重复读段的比例","唯一比对的读段数","被过滤掉读段数的比例","X染色体浓度","孕妇BMI","年龄","GW","身高","体重"]
# Clean: keep rows where these not null
df_cls = df_f[feature_list + ["abnormal_bin"]].dropna()
print("\nClassification dataset shape after dropna:", df_cls.shape)
X = df_cls[feature_list]
y = df_cls["abnormal_bin"]

# Scale for some models
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# imbalance -> use class_weight balanced, also try SMOTE
from collections import Counter
print(Counter(y))

# Train_test split
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve, average_precision_score, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train {X_train.shape} test {X_test.shape}")

models = {}
# Logistic
log = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
log.fit(X_train, y_train)
pred_log = log.predict(X_test)
prob_log = log.predict_proba(X_test)[:,1]
auc_log = roc_auc_score(y_test, prob_log)
print(f"\nLogistic AUC {auc_log:.3f}")
print(classification_report(y_test, pred_log))
models['Logistic'] = (log, prob_log, auc_log)

# Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
prob_rf = rf.predict_proba(X_test)[:,1]
auc_rf = roc_auc_score(y_test, prob_rf)
print(f"RF AUC {auc_rf:.3f}")
print(classification_report(y_test, rf.predict(X_test)))
models['RF'] = (rf, prob_rf, auc_rf)

# XGBoost
xclf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight= (len(y_train)-sum(y_train))/sum(y_train), random_state=42, n_jobs=-1, eval_metric='logloss')
xclf.fit(X_train, y_train)
prob_xgb = xclf.predict_proba(X_test)[:,1]
auc_xgb = roc_auc_score(y_test, prob_xgb)
print(f"XGB AUC {auc_xgb:.3f}")
print(classification_report(y_test, xclf.predict(X_test)))
models['XGB'] = (xclf, prob_xgb, auc_xgb)

# SVM
svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
svm.fit(X_train, y_train)
prob_svm = svm.predict_proba(X_test)[:,1]
auc_svm = roc_auc_score(y_test, prob_svm)
print(f"SVM AUC {auc_svm:.3f}")
print(classification_report(y_test, svm.predict(X_test)))
models['SVM'] = (svm, prob_svm, auc_svm)

# Cross-validated AUC for robustness (5-fold)
print("\n--- 5-fold CV AUC ---")
for name, (mdl, _, _) in models.items():
    # use appropriate model cloned? just cross_validate with same params
    from sklearn.base import clone
    # For RF, XGB etc need to clone original estimator type
    if name=='Logistic':
        est = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
    elif name=='RF':
        est = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    elif name=='XGB':
        est = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight= (len(y)-sum(y))/sum(y), random_state=42, n_jobs=-1, eval_metric='logloss')
    else:
        est = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    # scale inside CV? We'll use pipeline conceptually: scale first then train, but for CV we need to use scaled X
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs=[]
    for tr, te in skf.split(X_scaled, y):
        est_clone = clone(est)
        est_clone.fit(X_scaled[tr], y.iloc[tr])
        prob = est_clone.predict_proba(X_scaled[te])[:,1]
        aucs.append(roc_auc_score(y.iloc[te], prob))
    print(f"{name} CV AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f}")

# Feature importance for RF and XGB
imp_rf = pd.Series(rf.feature_importances_, index=feature_list).sort_values(ascending=False)
print("\nRF feature importance:")
print(imp_rf.round(4))
imp_xgb = pd.Series(xclf.feature_importances_, index=feature_list).sort_values(ascending=False)
print("\nXGB feature importance:")
print(imp_xgb.round(4))

plt.figure(figsize=(8,6))
imp_rf.sort_values().plot(kind='barh')
plt.title("RF Feature Importance (Female Abnormal)")
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q4_rf_importance.png", dpi=200)
plt.figure(figsize=(8,6))
imp_xgb.sort_values().plot(kind='barh')
plt.title("XGB Feature Importance (Female Abnormal)")
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q4_xgb_importance.png", dpi=200)

# ROC curves
plt.figure(figsize=(6,5))
for name, (_, prob, auc) in models.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Female Abnormal Detection")
plt.legend(); plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q4_roc.png", dpi=200)

# Precision Recall
plt.figure(figsize=(6,5))
for name, (_, prob, _) in models.items():
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    plt.plot(rec, prec, label=f"{name} AP={ap:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend(); plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q4_pr.png", dpi=200)

# Confusion matrix for best model (XGB probably)
# Find best
best_name = max(models, key=lambda k: models[k][2])
print(f"\nBest model: {best_name}")
best_prob = models[best_name][1]
# Tune threshold to maximize Youden or F1? Try threshold 0.5 default vs optimal via CV
from sklearn.metrics import f1_score
thresholds = np.linspace(0.1,0.9,81)
f1s = []
for th in thresholds:
    pred = (best_prob>=th).astype(int)
    f1s.append(f1_score(y_test, pred))
best_th = thresholds[np.argmax(f1s)]
print(f"Best threshold for F1: {best_th:.2f} F1={max(f1s):.3f}")
pred_best = (best_prob>=best_th).astype(int)
print(confusion_matrix(y_test, pred_best))
print(classification_report(y_test, pred_best))

# Additional analysis: Z-score threshold theory
# Typically |Z|>3 indicates aneuploidy. Check how well Z alone predicts
for col in ["13号染色体的Z值","18号染色体的Z值","21号染色体的Z值"]:
    # For each, if |Z|>3 predict abnormal of that type? But AB includes any.
    # Simple rule: max abs Z >3 => abnormal
    pass
# Evaluate simple threshold rule
df_cls["max_absZ"] = df_cls[["13号染色体的Z值","18号染色体的Z值","21号染色体的Z值"]].abs().max(axis=1)
# Find optimal threshold via ROC
from sklearn.metrics import roc_auc_score
auc_simple = roc_auc_score(y, df_cls["max_absZ"])
print(f"\nSimple max|Z| AUC {auc_simple:.3f}")
fpr_s, tpr_s,_ = roc_curve(y, df_cls["max_absZ"])
plt.figure(figsize=(5,4))
plt.plot(fpr_s, tpr_s, label=f"max|Z| AUC={auc_simple:.3f}")
plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title("Simple Z threshold ROC")
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q4_simpleZ_roc.png", dpi=200)

# Also check X chromosome concentration / Z for female abnormal? Might be related to maternal mosaicism?
# Save results
df_f[feature_list + ["abnormal_bin"]].to_csv("/home/user/test_arena/solution/results/q4_features.csv", index=False, encoding='utf-8-sig')

# Summary table: univariate AUC per feature
uni_aucs = {}
for feat in feature_list:
    try:
        # Use absolute Z for Z features? keep raw
        auc = roc_auc_score(y, X[feat])
        # also try abs
        if "Z值" in feat:
            auc_abs = roc_auc_score(y, X[feat].abs())
            uni_aucs[feat] = max(auc, auc_abs)
        else:
            uni_aucs[feat] = auc
    except:
        uni_aucs[feat] = np.nan
uni_series = pd.Series(uni_aucs).sort_values(ascending=False)
print("\nUnivariate AUC per feature:")
print(uni_series.round(3))

print("Q4 done")
