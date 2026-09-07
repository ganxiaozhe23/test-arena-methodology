import pandas as pd, numpy as np, re, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.rcParams['font.sans-serif']=['DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

# Load
df_m = pd.read_csv("/home/user/test_arena/男胎检测数据.csv", encoding='utf-8-sig')
df_f = pd.read_csv("/home/user/test_arena/女胎检测数据.csv", encoding='utf-8-sig')
def parse_gw(s):
    m=re.match(r"(\d+)w(?:\+(\d+))?", str(s).strip())
    if not m: return np.nan
    w=int(m.group(1)); d=int(m.group(2)) if m.group(2) else 0
    return w + d/7
df_m["孕周数值"] = df_m["检测孕周"].apply(parse_gw)
df_f["孕周数值"] = df_f["检测孕周"].apply(parse_gw)

# Quick info
print("MALE shape:", df_m.shape)
print("FEMALE shape:", df_f.shape)
print(df_m[["年龄","身高","体重","孕妇BMI","孕周数值","Y染色体浓度","原始读段数","GC含量"]].describe().to_string())
print("\nCorrelation matrix (male):")
cols_q1 = ["孕周数值","孕妇BMI","年龄","身高","体重","Y染色体浓度"]
print(df_m[cols_q1].corr().round(3))

# Check duplicates and repeated measures handling
print("\nMultiple measurements per pregnant code - male:")
print(df_m.groupby("孕妇代码").size().value_counts().head())
# Check distribution of Y conc by GW bin
df_m["达标"] = (df_m["Y染色体浓度"]>=0.04).astype(int)
print("\nOverall达标比例:", df_m["达标"].mean())
print(df_m.groupby(pd.cut(df_m["孕周数值"], bins=[10,12,15,20,25,30]))["达标"].mean())
print(df_m.groupby(pd.cut(df_m["孕妇BMI"], bins=[20,28,32,36,40,50]))["达标"].mean())

# --- Figure 1: scatter GW vs Y colored by BMI ---
fig, ax = plt.subplots(figsize=(7,5))
sc = ax.scatter(df_m["孕周数值"], df_m["Y染色体浓度"], c=df_m["孕妇BMI"], cmap="coolwarm", alpha=0.6, s=18)
plt.colorbar(sc, label="BMI")
ax.axhline(0.04, color='red', linestyle='--', label='Threshold 4%')
ax.set_xlabel("Gestational Week (weeks)")
ax.set_ylabel("Fetal Y-chromosome Concentration")
ax.set_title("Q1: Y-concentration vs Gestational Week (colored by BMI)")
ax.legend()
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q1_scatter_gw_y_bmi.png", dpi=200)

# Figure 2: BMI vs Y
fig, ax = plt.subplots(figsize=(7,5))
sc2 = ax.scatter(df_m["孕妇BMI"], df_m["Y染色体浓度"], c=df_m["孕周数值"], cmap="viridis", alpha=0.6, s=18)
plt.colorbar(sc2, label="Gestational Week")
ax.axhline(0.04, color='red', linestyle='--')
ax.set_xlabel("BMI (kg/m^2)")
ax.set_ylabel("Y-concentration")
ax.set_title("Q1: Y-concentration vs BMI (colored by GW)")
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q1_scatter_bmi_y.png", dpi=200)

# Figure 3: correlation heatmap
plt.figure(figsize=(7,6))
corr = df_m[["孕周数值","孕妇BMI","年龄","身高","体重","Y染色体浓度","原始读段数","GC含量","被过滤掉读段数的比例","X染色体浓度"]].corr()
sns.heatmap(corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f")
plt.title("Correlation Heatmap (Male)")
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q1_corr_heatmap.png", dpi=200)

# --- Q1 modeling ---
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

# clean : drop rows where Y is missing (should be none)
df_q1 = df_m.dropna(subset=["Y染色体浓度","孕周数值","孕妇BMI","年龄","身高","体重"]).copy()
# Add derived: BMI maybe computed from height weight, but use given; also add interaction
df_q1["BMI_GW"] = df_q1["孕妇BMI"] * df_q1["孕周数值"]
df_q1["GW2"] = df_q1["孕周数值"]**2
df_q1["BMI2"] = df_q1["孕妇BMI"]**2
df_q1["invBMI"] = 1/df_q1["孕妇BMI"]

# Model A: simple linear: Y ~ GW + BMI
X_a = sm.add_constant(df_q1[["孕周数值","孕妇BMI"]])
y = df_q1["Y染色体浓度"]
model_a = sm.OLS(y, X_a).fit()
print("\n=== Model A: Y ~ GW + BMI (OLS) ===")
print(model_a.summary())

# Model B: Y ~ GW + BMI + GW*BMI
X_b = sm.add_constant(df_q1[["孕周数值","孕妇BMI","BMI_GW"]])
model_b = sm.OLS(y, X_b).fit()
print("\n=== Model B: + interaction ===")
print(model_b.summary())

# Model C: Y ~ GW + BMI + GW2 + BMI2 + interaction (quadratic)
X_c = sm.add_constant(df_q1[["孕周数值","孕妇BMI","GW2","BMI2","BMI_GW"]])
model_c = sm.OLS(y, X_c).fit()
print("\n=== Model C: quadratic ===")
print(model_c.summary())

# Model D: log(Y) ~ GW + BMI (since Y maybe exponential growth? Check)
df_q1["logY"] = np.log(df_q1["Y染色体浓度"].clip(lower=1e-4))
X_d = sm.add_constant(df_q1[["孕周数值","孕妇BMI"]])
model_d = sm.OLS(df_q1["logY"], X_d).fit()
print("\n=== Model D: logY ~ GW + BMI ===")
print(model_d.summary())

# Model E: multiple with age height weight GC etc (full)
features_e = ["孕周数值","孕妇BMI","年龄","身高","体重","GC含量","原始读段数","被过滤掉读段数的比例"]
# standardize? but OLS can handle
X_e = sm.add_constant(df_q1[features_e])
model_e = sm.OLS(y, X_e).fit()
print("\n=== Model E: full covariates ===")
print(model_e.summary())

# Save summary to file
with open("/home/user/test_arena/solution/results/q1_model_summaries.txt","w",encoding="utf-8") as f:
    f.write("Model A summary\n"); f.write(str(model_a.summary())); f.write("\n\n")
    f.write("Model B\n"); f.write(str(model_b.summary())); f.write("\n\n")
    f.write("Model C\n"); f.write(str(model_c.summary())); f.write("\n\n")
    f.write("Model D logY\n"); f.write(str(model_d.summary())); f.write("\n\n")
    f.write("Model E full\n"); f.write(str(model_e.summary()))

# Cross-validated comparison
from sklearn.model_selection import KFold
def cv_r2(X, y):
    kf=KFold(5, shuffle=True, random_state=42)
    scores=[]
    for tr, te in kf.split(X):
        lr=LinearRegression().fit(X.iloc[tr], y.iloc[tr])
        pred=lr.predict(X.iloc[te])
        scores.append(r2_score(y.iloc[te], pred))
    return np.mean(scores), np.std(scores)

print("CV R2 Model A:", cv_r2(df_q1[["孕周数值","孕妇BMI"]], y))
print("CV R2 Model B:", cv_r2(df_q1[["孕周数值","孕妇BMI","BMI_GW"]], y))
print("CV R2 Model C:", cv_r2(df_q1[["孕周数值","孕妇BMI","GW2","BMI2","BMI_GW"]], y))
# RF and XGB
X_rf = df_q1[["孕周数值","孕妇BMI","年龄","身高","体重","GC含量"]]

from sklearn.preprocessing import StandardScaler
rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
# 5-fold CV
from sklearn.model_selection import cross_validate
cv_rf = cross_validate(rf, X_rf, y, cv=5, scoring=['r2','neg_mean_squared_error'])
print("RF CV R2:", cv_rf['test_r2'].mean(), "RMSE", np.sqrt(-cv_rf['test_neg_mean_squared_error'].mean()))

# XGBoost
xreg = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
cv_xgb = cross_validate(xreg, X_rf, y, cv=5, scoring=['r2','neg_mean_squared_error'])
print("XGB CV R2:", cv_xgb['test_r2'].mean())

# Residual diagnostics figure for Model A
fig, axes = plt.subplots(1,3,figsize=(12,3.5))
resid = model_a.resid
fitted = model_a.fittedvalues
axes[0].scatter(fitted, resid, alpha=0.5, s=12)
axes[0].axhline(0,color='red',linestyle='--')
axes[0].set_xlabel("Fitted")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs Fitted (Model A)")
# Q-Q
stats.probplot(resid, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q plot")
# histogram
axes[2].hist(resid, bins=30, edgecolor='black', alpha=0.7)
axes[2].set_title("Residual Histogram")
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q1_residual_diagnostics.png", dpi=200)

# Partial regression plots?
# ANOVA significance?
from scipy.stats import f_oneway, shapiro, levene
print("\nShapiro on residuals Model A p:", shapiro(resid)[1])
print("Levene test for达标 vs non-达标 Y variance:", levene(df_q1[df_q1["达标"]==1]["Y染色体浓度"], df_q1[df_q1["达标"]==0]["Y染色体浓度"]))

# Heteroscedasticity Breusch-Pagan
import statsmodels.stats.api as sms
bp_test = sms.het_breuschpagan(resid, X_a)
print("Breusch-Pagan heteroskedasticity test p:", bp_test[1])

# Save results csv
df_q1[["孕妇代码","检测孕周","孕周数值","孕妇BMI","Y染色体浓度","达标"]].to_csv("/home/user/test_arena/solution/results/q1_cleaned_male.csv", index=False, encoding='utf-8-sig')

print("Q1 figures saved, model summaries written")
