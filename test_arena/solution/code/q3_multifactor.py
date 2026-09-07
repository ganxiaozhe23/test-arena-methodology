import pandas as pd, numpy as np, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
warnings = __import__('warnings'); warnings.filterwarnings('ignore')

df_m = pd.read_csv("/home/user/test_arena/男胎检测数据.csv", encoding='utf-8-sig')
def parse_gw(s):
    m=re.match(r"(\d+)w(?:\+(\d+))?", str(s).strip())
    if m:
        w=int(m.group(1)); d=int(m.group(2)) if m.group(2) else 0
        return w+d/7
    return np.nan
df_m["GW"] = df_m["检测孕周"].apply(parse_gw)
df_m["BMI"] = df_m["孕妇BMI"]
df_m["Y"] = df_m["Y染色体浓度"]
df_m["pass"] = (df_m["Y"]>=0.04).astype(int)
df_m["age"] = df_m["年龄"]
df_m["height"] = df_m["身高"]
df_m["weight"] = df_m["体重"]
df_m["GC"] = df_m["GC含量"]
df_m["reads"] = df_m["原始读段数"]
df_m["filt"] = df_m["被过滤掉读段数的比例"]
df_m = df_m.dropna(subset=["GW","BMI","Y","age","height","weight","GC"])

# Feature importance via Random Forest for Y regression and pass classification
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.inspection import permutation_importance

features = ["GW","BMI","age","height","weight","GC","reads","filt"]
X = df_m[features]
y_reg = df_m["Y"]
y_clf = df_m["pass"]

# Regression RF
rf_reg = RandomForestRegressor(n_estimators=500, max_depth=8, random_state=42, n_jobs=-1)
rf_reg.fit(X, y_reg)
imp_reg = pd.Series(rf_reg.feature_importances_, index=features).sort_values(ascending=False)
print("RF Regressor Feature Importance (Y):")
print(imp_reg.round(4))

# Classification RF
rf_clf = RandomForestClassifier(n_estimators=500, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
rf_clf.fit(X, y_clf)
imp_clf = pd.Series(rf_clf.feature_importances_, index=features).sort_values(ascending=False)
print("\nRF Classifier Feature Importance (pass):")
print(imp_clf.round(4))

# Permutation importance more reliable
perm_reg = permutation_importance(rf_reg, X, y_reg, n_repeats=10, random_state=42)
perm_clf = permutation_importance(rf_clf, X, y_clf, n_repeats=10, random_state=42, scoring='roc_auc')
print("\nPermutation Importance Reg:")
for i in perm_reg.importances_mean.argsort()[::-1]:
    print(f"{features[i]}: {perm_reg.importances_mean[i]:.4f} +- {perm_reg.importances_std[i]:.4f}")
print("\nPermutation Importance Clf:")
for i in perm_clf.importances_mean.argsort()[::-1]:
    print(f"{features[i]}: {perm_clf.importances_mean[i]:.4f} +- {perm_clf.importances_std[i]:.4f}")

# SHAP approximation via feature importance plot
plt.figure(figsize=(7,4))
imp_reg.plot(kind='barh')
plt.title("RF Regression Feature Importance for Y")
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q3_imp_reg.png", dpi=200)
plt.figure(figsize=(7,4))
imp_clf.plot(kind='barh')
plt.title("RF Classification Feature Importance for pass")
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q3_imp_clf.png", dpi=200)

# Now multifactor grouping: use Decision Tree classifier/regressor to split BMI but considering other factors
# Approach: train decision tree regressor to predict Y or pass probability using BMI as primary split variable, but allow other splits
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text, plot_tree
# Use tree with max_leaf_nodes = number of groups we want (e.g., 4 groups)
for max_leaf in [4,5]:
    tree_reg = DecisionTreeRegressor(max_leaf_nodes=max_leaf, min_samples_leaf=20, random_state=42)
    tree_reg.fit(X, y_reg)
    print(f"\n=== Decision Tree Regressor max_leaf {max_leaf} ===")
    print(export_text(tree_reg, feature_names=features, max_depth=4))
    # plot
    plt.figure(figsize=(12,6))
    plot_tree(tree_reg, feature_names=features, filled=True, rounded=True, fontsize=8)
    plt.title(f"Decision Tree Regressor (Y) leaf={max_leaf}")
    plt.tight_layout()
    plt.savefig(f"/home/user/test_arena/solution/results/figures/q3_tree_reg_leaf{max_leaf}.png", dpi=200)
    
    tree_clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf, min_samples_leaf=20, class_weight='balanced', random_state=42)
    tree_clf.fit(X, y_clf)
    print(f"\n=== Decision Tree Classifier max_leaf {max_leaf} ===")
    print(export_text(tree_clf, feature_names=features, max_depth=4))
    plt.figure(figsize=(12,6))
    plot_tree(tree_clf, feature_names=features, filled=True, rounded=True, fontsize=8)
    plt.title(f"Decision Tree Classifier (pass) leaf={max_leaf}")
    plt.tight_layout()
    plt.savefig(f"/home/user/test_arena/solution/results/figures/q3_tree_clf_leaf{max_leaf}.png", dpi=200)

# Another approach: KMeans on multifactor space (BMI, weight, age, height) to get grouping, then analyze each cluster's optimal GW
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Use features that are significant: GW excluded for grouping? grouping should be based on maternal characteristics only, not GW.
group_features = ["BMI","weight","height","age"]  # GC, reads are technical not grouping
X_group = df_m[group_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_group)

for k in [3,4,5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"\nMultifactor KMeans k={k} silhouette={sil:.3f}")
    # Analyze each cluster's pass rate and T_star
    df_m["cluster_mult"] = labels
    for cl in range(k):
        sub = df_m[df_m["cluster_mult"]==cl]
        pass_rate = sub["pass"].mean()
        print(f"  Cluster {cl}: n={len(sub)} BMI mean {sub['BMI'].mean():.1f} weight {sub['weight'].mean():.1f} age {sub['age'].mean():.1f} pass_rate {pass_rate:.3f}")
        # T_star
        # need earliest time function similar to q2 but per cluster
        def earliest(sub_df):
            times=[]
            for code,g in sub_df.groupby("孕妇代码"):
                g=g.sort_values("GW")
                hits=g["pass"].values; gws=g["GW"].values; ys=g["Y"].values
                if 1 in hits:
                    idx=np.where(hits==1)[0][0]
                    if idx==0: times.append(gws[0])
                    else:
                        x0,y0=gws[idx-1],ys[idx-1]; x1,y1=gws[idx],ys[idx]
                        if y1!=y0:
                            t=x0+(0.04-y0)*(x1-x0)/(y1-y0)
                            times.append(np.clip(t,x0,x1))
                        else: times.append(gws[idx])
                else: times.append(np.nan)
            return np.array(times)
        t = earliest(sub)
        tv=t[~np.isnan(t)]
        if len(tv)>0:
            print(f"    T_star mean {np.nanmean(tv):.2f} median {np.nanmedian(tv):.2f}")

# Detailed per-pregnant analysis: find T_star per pregnant, then model T_star ~ BMI + other factors (multiple regression)
# Build dataset of per-pregnant earliest time and maternal factors
def compute_T_star(df):
    rows=[]
    for code,g in df.groupby("孕妇代码"):
        g=g.sort_values("GW")
        # maternal factors are constant per code (take first)
        bmi=g["BMI"].iloc[0]  # but BMI changes slightly with weight over GW; use mean
        bmi_mean=g["BMI"].mean()
        age=g["age"].iloc[0]
        height=g["height"].iloc[0]
        weight_mean=g["weight"].mean()
        GC_mean=g["GC"].mean()
        # compute T_star
        hits=g["pass"].values; gws=g["GW"].values; ys=g["Y"].values
        if 1 in hits:
            idx=np.where(hits==1)[0][0]
            if idx==0:
                t=gws[0]
            else:
                x0,y0=gws[idx-1],ys[idx-1]; x1,y1=gws[idx],ys[idx]
                t=x0+(0.04-y0)*(x1-x0)/(y1-y0) if y1!=y0 else gws[idx]
            rows.append([code,bmi_mean,age,height,weight_mean,GC_mean,t,1])
        else:
            # censored: record max GW as lower bound
            rows.append([code,bmi_mean,age,height,weight_mean,GC_mean,g["GW"].max(),0])
    return pd.DataFrame(rows, columns=["code","BMI","age","height","weight","GC","T_star","observed"])

df_T = compute_T_star(df_m)
print("\nPer-pregnant T_star dataset shape:", df_T.shape)
print(df_T.describe().round(2))
# Regression for T_star using only observed (not censored) first, then also consider Tobit/censored regression
import statsmodels.api as sm
# OLS for T_star ~ BMI + age + height + weight
df_obs = df_T[df_T["observed"]==1]
X_T = sm.add_constant(df_obs[["BMI","age","height","weight"]])
y_T = df_obs["T_star"]
model_T = sm.OLS(y_T, X_T).fit()
print("\nModel: T_star ~ BMI + age + height + weight (observed only, OLS)")
print(model_T.summary())

# Check multicollinearity VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["feature"] = X_T.columns
vif["VIF"] = [variance_inflation_factor(X_T.values, i) for i in range(X_T.shape[1])]
print("\nVIF:")
print(vif)

# Also try model with BMI only to compare R2
model_T_bmi = sm.OLS(y_T, sm.add_constant(df_obs[["BMI"]])).fit()
print("\nBMI only model:")
print(model_T_bmi.summary())

# Try XGBoost for T_star prediction
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
X_xgb = df_obs[["BMI","age","height","weight","GC"]]
y_xgb = df_obs["T_star"]
xreg = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
scores = cross_val_score(xreg, X_xgb, y_xgb, cv=5, scoring='r2')
print(f"\nXGBoost for T_star CV R2: {scores.mean():.3f} +- {scores.std():.3f}")

# Also analyze pass proportion at different GW thresholds per multifactor cluster
# Define达标比例 at proposed time
print("\n--- Q3 grouping with BMI+other factors, optimal timing considering pass proportion ---")
# Use 4-cluster KMeans as grouping
k=4
km4 = KMeans(n_clusters=k, random_state=42, n_init=20)
labels4 = km4.fit_predict(StandardScaler().fit_transform(df_m[group_features]))
df_m["grp_q3"] = labels4
# For each group, evaluate pass proportion vs GW and find optimal GW achieving at least 80% pass
import statsmodels.api as sm
for cl in range(k):
    sub = df_m[df_m["grp_q3"]==cl]
    print(f"\nGroup {cl}: BMI {sub['BMI'].mean():.1f} ±{sub['BMI'].std():.1f}, weight {sub['weight'].mean():.0f}, height {sub['height'].mean():.0f}, age {sub['age'].mean():.1f}, n_preg {sub['孕妇代码'].nunique()}, pass_rate overall {sub['pass'].mean():.3f}")
    # fit logistic pass ~ GW within group
    if sub["pass"].nunique()<2:
        print("  Not enough variation")
        continue
    Xl = sm.add_constant(sub["GW"])
    yl = sub["pass"]
    try:
        m = sm.Logit(yl, Xl).fit(disp=0)
        # grid GW 11-25, find smallest GW where p>=0.8 and also risk minimized
        gw_grid = np.linspace(11,25,140)
        probs = m.predict(sm.add_constant(gw_grid))
        # find GW where prob >=0.8
        idx_80 = np.where(probs>=0.8)[0]
        gw_80 = gw_grid[idx_80[0]] if len(idx_80)>0 else np.nan
        idx_90 = np.where(probs>=0.9)[0]
        gw_90 = gw_grid[idx_90[0]] if len(idx_90)>0 else np.nan
        print(f"  Logistic GW for p>=0.8: {gw_80:.2f}, p>=0.9: {gw_90:.2f}, coef GW {m.params['GW']:.3f} p={m.pvalues['GW']:.2e}")
        # also empirical T_star
        def earliest2(sdf):
            tt=[]
            for _,g in sdf.groupby("孕妇代码"):
                g=g.sort_values("GW")
                h=g["pass"].values; gws=g["GW"].values; ys=g["Y"].values
                if 1 in h:
                    idx=np.where(h==1)[0][0]
                    if idx==0: tt.append(gws[0])
                    else:
                        x0,y0=gws[idx-1],ys[idx-1]; x1,y1=gws[idx],ys[idx]
                        tt.append(x0+(0.04-y0)*(x1-x0)/(y1-y0) if y1!=y0 else gws[idx])
                    # clipped
                else: tt.append(np.nan)
            return np.array(tt)
        tt=earliest2(sub)
        print(f"  Empirical T_star median {np.nanmedian(tt):.2f} mean {np.nanmean(tt):.2f} 80pct {np.nanpercentile(tt[~np.isnan(tt)],80):.2f} 90pct {np.nanpercentile(tt[~np.isnan(tt)],90):.2f}")
    except Exception as e:
        print(f"  error {e}")

# --- Detection error analysis for Q3 (same as Q2 but with multifactor) ---
sigma_e=0.015
np.random.seed(42)
print("\n--- MC error analysis for Q3 groups (k=4 multifactor) ---")
for cl in range(k):
    sub=df_m[df_m["grp_q3"]==cl]
    # compute orig optimal GW (using logistic p>=0.8)
    # quick: use median T
    def median_T(sdf):
        tt=[]
        for _,g in sdf.groupby("孕妇代码"):
            g=g.sort_values("GW")
            h=g["pass"].values; gws=g["GW"].values; ys=g["Y"].values
            if 1 in h:
                idx=np.where(h==1)[0][0]
                if idx==0: tt.append(gws[0])
                else:
                    x0,y0=gws[idx-1],ys[idx-1]; x1,y1=gws[idx],ys[idx]
                    tt.append(x0+(0.04-y0)*(x1-x0)/(y1-y0) if y1!=y0 else gws[idx])
                #
            else: tt.append(np.nan)
        tt=np.array(tt)
        return np.nanmedian(tt)
    orig=median_T(sub)
    mc_vals=[]
    for it in range(300):
        sub_n=sub.copy()
        sub_n["Y"]=sub["Y"]+np.random.normal(0,sigma_e,len(sub))
        sub_n["pass"]=(sub_n["Y"]>=0.04).astype(int)
        mc_vals.append(median_T(sub_n))
    print(f"Group {cl} orig median {orig:.2f} MC {np.mean(mc_vals):.2f} +- {np.std(mc_vals):.2f}")

print("Q3 done")
