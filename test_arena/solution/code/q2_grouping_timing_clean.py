import pandas as pd, numpy as np, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

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
df_m = df_m.dropna(subset=["GW","BMI"])

def earliest_time(sub_df):
    times=[]
    for code, g in sub_df.groupby("孕妇代码"):
        g=g.sort_values("GW")
        hits=g["pass"].values
        gws=g["GW"].values
        ys=g["Y"].values
        if 1 in hits:
            idx=np.where(hits==1)[0][0]
            if idx==0:
                times.append(gws[0])
            else:
                x0,y0=gws[idx-1],ys[idx-1]
                x1,y1=gws[idx],ys[idx]
                if y1!=y0:
                    t=x0+(0.04-y0)*(x1-x0)/(y1-y0)
                    t=np.clip(t,x0,x1)
                    times.append(t)
                else:
                    times.append(gws[idx])
        else:
            times.append(np.nan)
    return np.array(times)

all_times=earliest_time(df_m)
print(f"Total pregnant {df_m['孕妇代码'].nunique()}, estimable {np.sum(~np.isnan(all_times))}, censored {np.sum(np.isnan(all_times))}")
print(f"T stats mean {np.nanmean(all_times):.2f} median {np.nanmedian(all_times):.2f} 25% {np.nanpercentile(all_times,25):.2f} 75% {np.nanpercentile(all_times,75):.2f}")

# Evaluate K
bmi_vals=df_m[["BMI"]].values
for k in [3,4,5,6]:
    kmeans=KMeans(n_clusters=k, random_state=42, n_init=10)
    labels=kmeans.fit_predict(bmi_vals)
    sil=silhouette_score(bmi_vals, labels)
    df_preg=df_m.groupby("孕妇代码").agg({"BMI":"mean"}).reset_index()
    preg_bmi=df_preg["BMI"].values.reshape(-1,1)
    kmeans_preg=KMeans(n_clusters=k, random_state=42, n_init=10).fit(preg_bmi)
    labels_preg=kmeans_preg.labels_
    groups=[]
    for lab in range(k):
        codes=df_preg.iloc[np.where(labels_preg==lab)[0]]["孕妇代码"]
        t=earliest_time(df_m[df_m["孕妇代码"].isin(codes)])
        t=t[~np.isnan(t)]
        groups.append(t)
    from scipy.stats import f_oneway
    try:
        F,p=f_oneway(*groups)
    except:
        F,p=np.nan,np.nan
    print(f"k={k} silhouette={sil:.3f} ANOVA F={F:.2f} p={p:.2e} centers {np.sort(kmeans.cluster_centers_.flatten()).round(2)}")

fixed_bins=[20,28,32,36,40,50]
labels_fixed=pd.cut(df_m["BMI"],bins=fixed_bins, right=False, include_lowest=True)
print("\nFixed bins distribution")
print(labels_fixed.value_counts().sort_index())
for i in range(len(fixed_bins)-1):
    low,high=fixed_bins[i],fixed_bins[i+1]
    sub=df_m[(df_m["BMI"]>=low)&(df_m["BMI"]<high)]
    t=earliest_time(sub)
    tv=t[~np.isnan(t)]
    pass_rate=sub["pass"].mean()
    print(f"Bin [{low},{high}): n_preg={sub['孕妇代码'].nunique()} n_meas={len(sub)} pass_rate={pass_rate:.3f} T mean {np.nanmean(tv):.2f} std {np.nanstd(tv):.2f} median {np.nanmedian(tv):.2f} censored {np.sum(np.isnan(t))}")

plt.figure(figsize=(7,4))
plt.hist(df_m["BMI"],bins=30,edgecolor='black',alpha=0.6)
for b in fixed_bins:
    plt.axvline(b,color='red',linestyle='--',alpha=0.6)
plt.xlabel("BMI");plt.ylabel("Count");plt.title("BMI Distribution with Fixed Bins")
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q2_bmi_hist_fixed.png",dpi=200)

q_bins=[0,0.2,0.4,0.6,0.8,1.0]
q_vals=df_m["BMI"].quantile(q_bins).values
print("\nQuantile bins:",q_vals)
plt.figure(figsize=(6,4))
sns.boxplot(x=pd.cut(df_m["BMI"],bins=q_vals,include_lowest=True), y=df_m["Y"])
plt.xticks(rotation=30);plt.title("Y by Quantile BMI groups")
plt.tight_layout()
plt.savefig("/home/user/test_arena/solution/results/figures/q2_y_by_quantile_bmi.png",dpi=200)

import statsmodels.api as sm
def find_optimal(sub_df, target_prob=0.85):
    if sub_df["pass"].nunique()<2 or len(sub_df)<20:
        t=earliest_time(sub_df)
        t=t[~np.isnan(t)]
        if len(t)==0:
            return np.nan,None
        return np.percentile(t,85),None
    X=sm.add_constant(sub_df["GW"])
    y=sub_df["pass"]
    try:
        model=sm.Logit(y,X).fit(disp=0)
        a,b=model.params["const"],model.params["GW"]
        if b<=0:
            t=earliest_time(sub_df)
            t=t[~np.isnan(t)]
            return np.percentile(t,85) if len(t)>0 else np.nan,model
        logit_p=np.log(target_prob/(1-target_prob))
        gw_star=(logit_p-a)/b
        return gw_star,model
    except:
        t=earliest_time(sub_df)
        t=t[~np.isnan(t)]
        return np.percentile(t,85) if len(t)>0 else np.nan,None

print("\n--- Optimal timing per fixed BMI bin (Logistic) ---")
for low,high in zip(fixed_bins[:-1],fixed_bins[1:]):
    sub=df_m[(df_m["BMI"]>=low)&(df_m["BMI"]<high)]
    for tp in [0.9,0.85,0.8]:
        gw,_=find_optimal(sub,tp)
        print(f"[{low},{high}) target {tp}: GW*={gw:.2f}" if not np.isnan(gw) else f"[{low},{high}) NA")

k=4
kmeans=KMeans(n_clusters=k,random_state=42,n_init=10)
df_preg=df_m.groupby("孕妇代码").agg({"BMI":"mean"}).reset_index()
preg_bmi_vals=df_preg["BMI"].values.reshape(-1,1)
kmeans.fit(preg_bmi_vals)
df_preg["cluster"]=kmeans.labels_
centers=kmeans.cluster_centers_.flatten()
order=np.argsort(centers)
label_map={old:new for new,old in enumerate(order)}
df_preg["cluster_ordered"]=df_preg["cluster"].map(label_map)
print("\nKmeans k=4 centers sorted:",np.sort(centers).round(2))
for cl in sorted(df_preg["cluster_ordered"].unique()):
    codes=df_preg[df_preg["cluster_ordered"]==cl]["孕妇代码"]
    sub=df_m[df_m["孕妇代码"].isin(codes)]
    gw,_=find_optimal(sub,0.9)
    print(f"Cluster {cl}: center {np.sort(centers)[cl]:.2f} range ({sub['BMI'].min():.1f}-{sub['BMI'].max():.1f}) n_preg={len(codes)} GW*0.9={gw:.2f if not np.isnan(gw) else 'NA'}")

def time_risk(t):
    if t<=12: return 0
    elif t<=27: return (t-12)/15*1.0
    else: return 1.0+(t-27)*0.5
def inacc_risk(p): return 1-p

w1,w2=1.0,2.0
print("\n--- Risk optimization per fixed bin (w1=1,w2=2) ---")
gw_grid=np.linspace(11,25,100)
for low,high in zip(fixed_bins[:-1],fixed_bins[1:]):
    sub=df_m[(df_m["BMI"]>=low)&(df_m["BMI"]<high)]
    if sub["pass"].nunique()<2: continue
    X=sm.add_constant(sub["GW"])
    y=sub["pass"]
    try:
        model=sm.Logit(y,X).fit(disp=0)
        risks=[];probs=[]
        for gw in gw_grid:
            p=model.predict([1,gw])[0]
            p=np.clip(p,0.001,0.999)
            r=w1*time_risk(gw)+w2*inacc_risk(p)
            risks.append(r);probs.append(p)
        idx=np.argmin(risks)
        print(f"[{low},{high}): opt GW={gw_grid[idx]:.2f} p={probs[idx]:.3f} risk={risks[idx]:.3f}")
        plt.figure(figsize=(6,4))
        plt.plot(gw_grid,probs,label='P(pass)')
        plt.plot(gw_grid,[time_risk(g) for g in gw_grid],label='Time risk')
        plt.plot(gw_grid,risks,label='Total risk')
        plt.axvline(gw_grid[idx],color='red',linestyle='--',label=f'Opt {gw_grid[idx]:.1f}w')
        plt.xlabel("Gestational Week");plt.ylabel("Risk / Prob")
        plt.title(f"BMI [{low},{high}) Risk Optimization")
        plt.legend();plt.tight_layout()
        plt.savefig(f"/home/user/test_arena/solution/results/figures/q2_risk_opt_{low}_{high}.png",dpi=200)
        plt.close()
    except Exception as e:
        print(f"[{low},{high}) error {e}")

sigma_e=0.015
np.random.seed(42)
print("\n--- Monte Carlo error sensitivity ---")
for low,high in zip(fixed_bins[:-1],fixed_bins[1:]):
    sub=df_m[(df_m["BMI"]>=low)&(df_m["BMI"]<high)]
    t_orig=earliest_time(sub)
    tv=t_orig[~np.isnan(t_orig)]
    if len(tv)==0: continue
    mean_orig=np.nanmean(tv)
    mc_means=[]
    for it in range(500):
        sub_n=sub.copy()
        sub_n["Y"]=sub["Y"]+np.random.normal(0,sigma_e,len(sub))
        sub_n["pass"]=(sub_n["Y"]>=0.04).astype(int)
        tmc=earliest_time(sub_n)
        tmc=tmc[~np.isnan(tmc)]
        if len(tmc)>0:
            mc_means.append(np.nanmean(tmc))
    if len(mc_means)>0:
        print(f"[{low},{high}): orig {mean_orig:.2f} MC mean {np.mean(mc_means):.2f} +-{np.std(mc_means):.2f} bias {np.mean(mc_means)-mean_orig:.2f}")
print("Q2 clean done")
