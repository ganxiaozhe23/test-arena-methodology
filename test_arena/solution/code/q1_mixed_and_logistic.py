import pandas as pd, numpy as np, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns, warnings
warnings.filterwarnings('ignore')
df_m = pd.read_csv("/home/user/test_arena/男胎检测数据.csv", encoding='utf-8-sig')
def parse_gw(s):
    m=re.match(r"(\d+)w(?:\+(\d+))?", str(s).strip())
    if m:
        w=int(m.group(1)); d=int(m.group(2)) if m.group(2) else 0
        return w+d/7
    return np.nan
df_m["孕周数值"] = df_m["检测孕周"].apply(parse_gw)
df_m["达标"] = (df_m["Y染色体浓度"]>=0.04).astype(int)
df_m = df_m.dropna(subset=["孕周数值","Y染色体浓度","孕妇BMI"])

import statsmodels.api as sm
import statsmodels.formula.api as smf

# --- MixedLM: Y ~ GW + BMI + random intercept per 孕妇代码 ---
print("Fitting MixedLM ...")
# Need to handle grouping
md = smf.mixedlm("Q('Y染色体浓度') ~ 孕周数值 + Q('孕妇BMI')", df_m, groups=df_m["孕妇代码"])
mdf = md.fit(reml=False)
print(mdf.summary())

# Add more covariates mixed
md2 = smf.mixedlm("Q('Y染色体浓度') ~ 孕周数值 + Q('孕妇BMI') + 年龄 + 身高 + 体重 + Q('GC含量')", df_m, groups=df_m["孕妇代码"])
try:
    mdf2 = md2.fit(reml=False, maxiter=200)
    print(mdf2.summary())
except Exception as e:
    print("MixedLM2 failed:", e)

# --- Logistic regression for 达标概率 ---
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
df_m["BMI_GW"] = df_m["孕妇BMI"]*df_m["孕周数值"]
logit_data = df_m[["达标","孕周数值","孕妇BMI","年龄","身高","体重"]]
# statsmodels logit
import statsmodels.api as sm
X_logit = sm.add_constant(df_m[["孕周数值","孕妇BMI"]])
y_logit = df_m["达标"]
logit_model = sm.Logit(y_logit, X_logit).fit(disp=0)
print("\n=== Logistic 达标 ~ GW + BMI ===")
print(logit_model.summary())
print("AUC:", roc_auc_score(y_logit, logit_model.predict(X_logit)))

# Full logit with all
X_logit2 = sm.add_constant(df_m[["孕周数值","孕妇BMI","年龄","身高","体重","GC含量"]].dropna())
y2 = df_m.loc[X_logit2.index, "达标"]
logit2 = sm.Logit(y2, X_logit2).fit(disp=0, maxiter=100)
print(logit2.summary())
print("AUC full:", roc_auc_score(y2, logit2.predict(X_logit2)))

# --- Beta regression concept? Use logit transform? ---
# Also Spearman correlation
from scipy.stats import spearmanr, pearsonr, kendalltau
for col in ["孕周数值","孕妇BMI","年龄","身高","体重","GC含量"]:
    pr, pp = pearsonr(df_m[col], df_m["Y染色体浓度"])
    sr, sp = spearmanr(df_m[col], df_m["Y染色体浓度"])
    print(f"{col}: Pearson r={pr:.3f} p={pp:.2e}, Spearman r={sr:.3f} p={sp:.2e}")

# --- Box-Cox transform to find best lambda ---
from scipy.stats import boxcox
y_pos = df_m["Y染色体浓度"].clip(lower=0.001)
y_bc, lam = boxcox(y_pos)
print(f"\nBox-Cox lambda for Y: {lam:.3f}")
# fit OLS on boxcox Y
import statsmodels.api as sm
X_bc = sm.add_constant(df_m[["孕周数值","孕妇BMI"]])
bc_model = sm.OLS(y_bc, X_bc).fit()
print(bc_model.summary())

# --- Within-subject slope analysis ---
# For each孕妇 with >=3 measurements, fit individual slope GW->Y
from collections import defaultdict
slopes=[]
for code, sub in df_m.groupby("孕妇代码"):
    if len(sub)>=3:
        # linear regression per subject
        X = sm.add_constant(sub["孕周数值"])
        try:
            m = sm.OLS(sub["Y染色体浓度"], X).fit()
            slopes.append(m.params["孕周数值"])
        except: pass
print(f"\nIndividual slopes (n={len(slopes)}): mean={np.mean(slopes):.5f}, std={np.std(slopes):.5f}, median={np.median(slopes):.5f}")
plt.figure(figsize=(6,4))
plt.hist(slopes, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(slopes), color='red', linestyle='--', label=f'Mean={np.mean(slopes):.4f}')
plt.xlabel("Individual slope (Y per week)")
plt.ylabel("Count")
plt.title("Distribution of Within-Subject Y vs GW Slopes")
plt.legend()
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q1_individual_slopes.png", dpi=200)

# --- Figure: logistic predicted probability vs GW for different BMI groups ---
bmi_levels = [28, 32, 36, 40]
gw_grid = np.linspace(11, 25, 100)
plt.figure(figsize=(7,5))
for bmi in bmi_levels:
    X_grid = pd.DataFrame({"const":1, "孕周数值":gw_grid, "孕妇BMI": bmi})
    X_grid = X_grid[["const","孕周数值","孕妇BMI"]]
    # use logit coefficients
    prob = logit_model.predict(X_grid)
    plt.plot(gw_grid, prob, label=f'BMI={bmi}')
plt.axhline(0.9, color='gray', linestyle=':', label='90%达标 threshold')
plt.axhline(0.5, color='black', linestyle=':', label='50%')
plt.xlabel("Gestational Week")
plt.ylabel("Predicted P(Y>=4%)")
plt.title("Logistic Model: P(达标) vs GW by BMI")
plt.legend()
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q1_logistic_prob.png", dpi=200)

print("Done q1 mixed")
