import pandas as pd, numpy as np, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
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
df_m = df_m.dropna(subset=["孕周数值","孕妇BMI"])

# Helper to compute earliest达标 time per pregnant woman via linear interpolation
def earliest_time_for_group(sub_df):
    # For each孕妇代码, find first达标 time via interpolation between last non-达标 and first达标
    times=[]
    for code, g in sub_df.groupby("孕妇代码"):
        g = g.sort_values("孕周数值")
        # if already达标 at first measurement, time is first GW
        # if never达标, consider censored (return max GW or np.nan)
        達 = g["达标"].values
        gws = g["孕周数值"].values
        ys = g["Y染色体浓度"].values
        if 1 in 達:
            idx_first = np.where(達==1)[0][0]
            if idx_first==0:
                times.append(gws[0])
            else:
                # interpolate between idx_first-1 and idx_first
                x0, y0 = gws[idx_first-1], ys[idx_first-1]
                x1, y1 = gws[idx_first], ys[idx_first]
                if y1!=y0:
                    t_star = x0 + (0.04 - y0)*(x1 - x0)/(y1 - y0)
                    # clip to interval
                    t_star = np.clip(t_star, x0, x1)
                    times.append(t_star)
                else:
                    times.append(gws[idx_first])
        else:
            # never达标: treat as >max GW, use max+? For optimization we penalize
            # Use max GW + 2 as proxy
            times.append(np.nan)  # censored
    times = np.array(times)
    # return not nan
    return times

# Explore: overall distribution of T_star
# For whole cohort
all_times = earliest_time_for_group(df_m)
print(f"All pregnant: total {df_m['孕妇代码'].nunique()}, have estimable T_star {np.sum(~np.isnan(all_times))}, censored {np.sum(np.isnan(all_times))}")
print(f"T_star stats (non-censored): mean={np.nanmean(all_times):.2f}, median={np.nanmedian(all_times):.2f}, 25%={np.nanpercentile(all_times,25):.2f}, 75%={np.nanpercentile(all_times,75):.2f}, min={np.nanmin(all_times):.2f}, max={np.nanmax(all_times):.2f}")

# BMI grouping candidates: evaluate K=3,4,5,6 via K-means on BMI only
bmi_vals = df_m[["孕妇BMI"]].values
scaler = StandardScaler()
# But for 1D K-means, just use raw BMI
best_k = None; best_score= -1
for k in [3,4,5,6]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(bmi_vals)
    # silhouette requires at least 2 clusters and less than n_samples
    try:
        sil = silhouette_score(bmi_vals, labels)
    except:
        sil = -1
    # ANOVA-like: between vs within variance of Y? Compute F statistic of T_star across groups
    # Build pseudo grouping by assigning each measurement's BMI to cluster
    # Better compute by pregnant-level mean BMI
    df_preg = df_m.groupby("孕妇代码").agg({"孕妇BMI":"mean", "孕周数值":"mean", "Y染色体浓度":"mean"}).reset_index()
    # map each preg's BMI to cluster? easier: use preg-level BMI
    preg_bmi = df_preg["孕妇BMI"].values.reshape(-1,1)
    kmeans_preg = KMeans(n_clusters=k, random_state=42, n_init=10).fit(preg_bmi)
    labels_preg = kmeans_preg.labels_
    # compute ANOVA for T_star per group
    groups = []
    for lab in range(k):
        codes_in_cluster = df_preg.iloc[np.where(labels_preg==lab)[0]]["孕妇代码"]
        times_cluster = earliest_time_for_group(df_m[df_m["孕妇代码"].isin(codes_in_cluster)])
        times_cluster = times_cluster[~np.isnan(times_cluster)]
        groups.append(times_cluster)
    # F-test via one-way ANOVA if enough data
    from scipy.stats import f_oneway
    try:
        F, p = f_oneway(*groups)
    except: F,p = np.nan, np.nan
    print(f"k={k}, silhouette={sil:.3f}, ANOVA F={F:.2f}, p={p:.2e}, cluster centers sorted: {np.sort(kmeans.cluster_centers_.flatten())}")

# Now evaluate natural breaks (Jenks) vs quantile vs given fixed bins [20,28),[28,32),[32,36),[36,40),[40,inf)
fixed_bins = [20,28,32,36,40,50]
labels_fixed = pd.cut(df_m["孕妇BMI"], bins=fixed_bins, labels=[f"{fixed_bins[i]}-{fixed_bins[i+1]}" for i in range(len(fixed_bins)-1)], right=False, include_lowest=True)
print("\nFixed bins distribution:")
print(labels_fixed.value_counts().sort_index())
# For each fixed bin, compute T_star stats
for i in range(len(fixed_bins)-1):
    low, high = fixed_bins[i], fixed_bins[i+1]
    sub = df_m[(df_m["孕妇BMI"]>=low) & (df_m["孕妇BMI"]< high)]
    times = earliest_time_for_group(sub)
    times_valid = times[~np.isnan(times)]
        达_rate = sub["达标"].mean()
        达_rate = sub["达标"].mean()

# Plot BMI histogram with clusters
plt.figure(figsize=(7,4))
plt.hist(df_m["孕妇BMI"], bins=30, edgecolor='black', alpha=0.6)
for b in fixed_bins:
    plt.axvline(b, color='red', linestyle='--', alpha=0.6)
plt.xlabel("BMI")
plt.ylabel("Count")
plt.title("BMI Distribution with Fixed Bins")
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q2_bmi_hist_fixed.png", dpi=200)

# Alternative: use quantile bins
q_bins = [0,0.2,0.4,0.6,0.8,1.0]
q_vals = df_m["孕妇BMI"].quantile(q_bins).values
print("\nQuantile bins:", q_vals)
plt.figure(figsize=(6,4))
sns.boxplot(x=pd.cut(df_m["孕妇BMI"], bins=q_vals, include_lowest=True), y=df_m["Y染色体浓度"])
plt.xticks(rotation=30)
plt.title("Y by Quantile BMI groups")
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q2_y_by_quantile_bmi.png", dpi=200)

# --- Estimate logistic model per group to find optimal timing ---
import statsmodels.api as sm
from scipy.stats import logistic

def find_optimal_time(sub_df, target_prob=0.85, risk_weights=None):
    # Fit logistic P(达标) ~ GW
    if sub_df["达标"].nunique()<2 or len(sub_df)<20:
        # not enough variation, fallback to empirical percentile
        times = earliest_time_for_group(sub_df)
        times = times[~np.isnan(times)]
        if len(times)==0:
            return np.nan, np.nan
        # 85th percentile as covering
        return np.percentile(times, 85), 0.5
    X = sm.add_constant(sub_df["孕周数值"])
    y = sub_df["达标"]
    try:
        model = sm.Logit(y, X).fit(disp=0)
        # solve for GW where prob = target_prob: logit(p)=a+b*GW => GW=(logit(p)-a)/b
        a, b = model.params["const"], model.params["孕周数值"]
        if b<=0:
            # GW increase should increase prob, if b negative, model weird
            times = earliest_time_for_group(sub_df)
            times = times[~np.isnan(times)]
            return np.percentile(times, 85) if len(times)>0 else np.nan, model
        logit_p = np.log(target_prob/(1-target_prob))
        gw_star = (logit_p - a)/b
        return gw_star, model
    except Exception as e:
        times = earliest_time_for_group(sub_df)
        times = times[~np.isnan(times)]
        return np.percentile(times, 85) if len(times)>0 else np.nan, None

# Evaluate for fixed bins
print("\n--- Optimal timing per fixed BMI bin (Logistic target p=0.90, 0.85, 0.80) ---")
for low, high in zip(fixed_bins[:-1], fixed_bins[1:]):
    sub = df_m[(df_m["孕妇BMI"]>=low) & (df_m["孕妇BMI"]< high)]
    for tp in [0.9,0.85,0.8]:
        gw_star,_ = find_optimal_time(sub, target_prob=tp)
        print(f"[{low},{high}) target {tp}: GW*={gw_star:.2f}" if not np.isnan(gw_star) else f"[{low},{high}) target {tp}: NA")

# Also try Kmeans with k=4 on BMI (maybe more interpretable)
k=4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
# Use per-pregnant mean BMI to define grouping, then map measurements
df_preg = df_m.groupby("孕妇代码").agg({"孕妇BMI":"mean"}).reset_index()
preg_bmi_vals = df_preg["孕妇BMI"].values.reshape(-1,1)
kmeans.fit(preg_bmi_vals)
df_preg["cluster"] = kmeans.labels_
# sort clusters by center
centers = kmeans.cluster_centers_.flatten()
order = np.argsort(centers)
# remap labels to ordered 0..k-1
label_map = {old:new for new, old in enumerate(order)}
df_preg["cluster_ordered"] = df_preg["cluster"].map(label_map)
print("\nKmeans k=4 centers sorted:", np.sort(centers))
for cl in sorted(df_preg["cluster_ordered"].unique()):
    codes = df_preg[df_preg["cluster_ordered"]==cl]["孕妇代码"]
    sub = df_m[df_m["孕妇代码"].isin(codes)]
    bmimean = sub["孕妇BMI"].mean()
    bmirange = (sub["孕妇BMI"].min(), sub["孕妇BMI"].max())
    gw_star,_ = find_optimal_time(sub, 0.9)
    print(f"Cluster {cl}: BMI center {np.sort(centers)[cl]:.2f}, range {bmirange}, n_preg={len(codes)}, n_meas={len(sub)}, GW*0.9={gw_star:.2f if not np.isnan(gw_star) else 'NA'}")

# Risk model: Cost(t)= late detection risk + inaccuracy risk
# Define risk scoring: if t<=12: risk_time=0, 13-27: risk_time=(t-12)/15 *1, >=28: risk_time=1 + (t-27)*2 (steeper)
def time_risk(t):
    if t<=12:
        return 0
    elif t<=27:
        return (t-12)/15 *1.0  # linearly 0->1
    else:
        return 1.0 + (t-27)*0.5  # escalate quickly

def inaccuracy_risk(p):
    return 1-p  # probability of non-达标

# For each BMI group, compute total risk = w1*time_risk + w2*inaccuracy_risk at grid GW
# Find GW that minimizes total risk (optimal trade-off)
w1, w2 = 1.0, 2.0  # weight inaccuracy more? tuning needed
print("\n--- Risk optimization per fixed bin (w1=1,w2=2) ---")
gw_grid = np.linspace(11, 25, 100)
for low, high in zip(fixed_bins[:-1], fixed_bins[1:]):
    sub = df_m[(df_m["孕妇BMI"]>=low) & (df_m["孕妇BMI"]< high)]
    # fit logistic for p(GW)
    if sub["达标"].nunique()<2:
        continue
    X = sm.add_constant(sub["孕周数值"])
    y = sub["达标"]
    try:
        model = sm.Logit(y, X).fit(disp=0)
        risks=[]
        probs=[]
        for gw in gw_grid:
            p = model.predict([1, gw])[0]
            # clip
            p = np.clip(p, 0.001, 0.999)
            r = w1*time_risk(gw) + w2*inaccuracy_risk(p)
            risks.append(r); probs.append(p)
        idx_opt = np.argmin(risks)
        print(f"[{low},{high}): opt GW={gw_grid[idx_opt]:.2f}, p={probs[idx_opt]:.3f}, risk={risks[idx_opt]:.3f}")
        # also save figure per group
        plt.figure(figsize=(6,4))
        plt.plot(gw_grid, probs, label='P(达标)')
        plt.plot(gw_grid, [time_risk(g) for g in gw_grid], label='Time risk')
        plt.plot(gw_grid, risks, label='Total risk')
        plt.axvline(gw_grid[idx_opt], color='red', linestyle='--', label=f'Opt {gw_grid[idx_opt]:.1f}w')
        plt.xlabel("Gestational Week")
        plt.ylabel("Risk / Prob")
        plt.title(f"BMI [{low},{high}) Risk Optimization")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"/home/user/test_arena/solution/results/figures/q2_risk_opt_{low}_{high}.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"[{low},{high}) error {e}")

# --- Monte Carlo error simulation ---
# Assume measurement error of Y conc ~ N(0, sigma_e), sigma from replicate variance?
# Estimate sigma_e from within-pregnant repeated measurements at similar GW? Use overall residual SD from Model A ~0.032
sigma_e = 0.015  # plausible: from residual std=0.032 but part is biological; assume technical error smaller
np.random.seed(42)
print("\n--- Monte Carlo detection error sensitivity (sigma_e=0.015) ---")
for low, high in zip(fixed_bins[:-1], fixed_bins[1:]):
    sub = df_m[(df_m["孕妇BMI"]>=low) & (df_m["孕妇BMI"]< high)]
    times_orig = earliest_time_for_group(sub)
    times_orig_valid = times_orig[~np.isnan(times_orig)]
    if len(times_orig_valid)==0:
        continue
    mean_orig = np.nanmean(times_orig_valid)
    # MC: add noise to Y and recompute
    mc_means=[]
    for iter_ in range(500):
        sub_noisy = sub.copy()
        sub_noisy["Y染色体浓度"] = sub["Y染色体浓度"] + np.random.normal(0, sigma_e, len(sub))
        sub_noisy["达标"] = (sub_noisy["Y染色体浓度"]>=0.04).astype(int)
        times_mc = earliest_time_for_group(sub_noisy)
        times_mc = times_mc[~np.isnan(times_mc)]
        if len(times_mc)>0:
            mc_means.append(np.nanmean(times_mc))
    if len(mc_means)>0:
        print(f"[{low},{high}): orig mean {mean_orig:.2f}, MC mean {np.mean(mc_means):.2f} ±{np.std(mc_means):.2f}, bias {np.mean(mc_means)-mean_orig:.2f}")

print("Q2 done")