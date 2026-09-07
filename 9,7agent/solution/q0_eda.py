# -*- coding: utf-8 -*-
"""EDA: 男胎 Y 染色体浓度与孕周/BMI/个体特征的探索性分析, 出图"""
import sys
sys.path.insert(0, "solution")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from scipy import stats
from common import load_male, Y_TH

# ---------- 中文字体 ----------
font_path = None
for cand in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    import os
    if os.path.exists(cand):
        font_path = cand
        break
if font_path:
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
OUT = "figs"
import os
os.makedirs(OUT, exist_ok=True)

m = load_male()
m = m.dropna(subset=["y_conc", "week", "bmi"]).copy()

# ============ 图1: 浓度分布 & 达标率 ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(m.y_conc, bins=50, color="#4C72B0", edgecolor="white")
axes[0].axvline(Y_TH, color="r", ls="--", lw=1.5, label=f"阈值 {Y_TH*100:.0f}%")
axes[0].set_xlabel("Y 染色体浓度"); axes[0].set_ylabel("样本数"); axes[0].set_title("(a) 浓度分布")
axes[0].legend()

bins = np.arange(10.5, 28.5, 1.5)
lab = [f"{b:.0f}-{b+1.5:.0f}" for b in bins[:-1]]
m["wk_bin"] = pd.cut(m.week, bins=bins, labels=lab)
rate = m.groupby("wk_bin", observed=True).apply(lambda d: pd.Series({
    "rate": (d.y_conc >= Y_TH).mean(), "n": len(d)}), include_groups=False).reset_index()
axes[1].plot(range(len(rate)), rate.rate, "-o", color="#55A868")
axes[1].set_xticks(range(len(rate))); axes[1].set_xticklabels(rate.wk_bin, rotation=45, fontsize=8)
axes[1].set_ylabel("达标率 (Y≥4%)"); axes[1].set_title("(b) 各孕周达标率")

bmibins = [20, 28, 32, 36, 40, np.inf]
mlab = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "40+"]
m["bmi_bin"] = pd.cut(m.bmi, bins=bmibins, labels=mlab, right=False)
r2 = m.groupby("bmi_bin", observed=True).apply(lambda d: pd.Series({
    "rate": (d.y_conc >= Y_TH).mean(), "n": len(d)}), include_groups=False).reset_index()
axes[2].plot(range(len(r2)), r2.rate, "-s", color="#C44E52")
axes[2].set_xticks(range(len(r2))); axes[2].set_xticklabels(r2.bmi_bin)
axes[2].set_ylabel("达标率"); axes[2].set_title("(c) 各 BMI 段达标率")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_dist.png"); plt.close()

# ============ 图2: 散点 Y~week (按BMI着色) & Y~BMI ============
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
sc = axes[0].scatter(m.week, m.y_conc, c=m.bmi, s=10, alpha=0.6, cmap="viridis")
axes[0].axhline(Y_TH, color="r", ls="--", lw=1)
axes[0].set_xlabel("孕周"); axes[0].set_ylabel("Y 染色体浓度"); axes[0].set_title("(a) 浓度 vs 孕周 (颜色=BMI)")
cb = plt.colorbar(sc, ax=axes[0]); cb.set_label("BMI")

for lbl, dd in m.groupby("bmi_bin", observed=True):
    axes[1].scatter(dd.week, dd.y_conc, s=8, alpha=0.5, label=lbl)
axes[1].axhline(Y_TH, color="r", ls="--", lw=1)
axes[1].set_xlabel("孕周"); axes[1].set_ylabel("Y 染色体浓度"); axes[1].set_title("(b) 按 BMI 分组")
axes[1].legend(fontsize=8)

for g, dd in m.groupby("bmi_bin", observed=True):
    if len(dd) < 5:
        continue
    sm = np.polyfit(dd.week, dd.y_conc, 1)
    xv = np.linspace(10.5, 28, 50)
    axes[2].plot(xv, np.polyval(sm, xv), label=f"{g} (slope={sm[0]:.4f})", lw=1.6)
axes[2].axhline(Y_TH, color="r", ls="--", lw=1)
axes[2].set_xlabel("孕周"); axes[2].set_ylabel("拟合 Y 浓度"); axes[2].set_title("(c) 各 BMI 段线性趋势")
axes[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_trend.png"); plt.close()

# ============ 图3: 同孕妇纵向轨迹样例 ============
fig, ax = plt.subplots(figsize=(8, 5))
for pid in m.pid.unique()[:24]:
    dd = m[m.pid == pid].sort_values("week")
    ax.plot(dd.week, dd.y_conc, "-o", ms=3, lw=0.8, alpha=0.7)
ax.axhline(Y_TH, color="r", ls="--", lw=1.2)
ax.set_xlabel("孕周"); ax.set_ylabel("Y 染色体浓度"); ax.set_title("部分孕妇纵向轨迹(同一孕妇多次检测)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_traj.png"); plt.close()

# ============ 图4: 相关矩阵 ============
cols = ["y_conc", "week", "bmi", "age", "height", "weight", "raw_reads", "gc",
        "uniq_reads", "x_conc", "map_ratio", "dup_ratio", "flt_ratio"]
cc = m[cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(9.5, 8))
sns.heatmap(cc, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            cbar_kws={"label": "Spearman ρ"}, ax=ax)
ax.set_title("男胎数据 Spearman 相关矩阵")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_corr.png"); plt.close()

# ============ 数值输出 ============
print("=== 与 Y 浓度的 Spearman/Pearson 相关(全样本 n=%d) ===" % len(m))
for c in ["week", "bmi", "age", "height", "weight", "raw_reads", "gc", "uniq_reads", "x_conc"]:
    rp = stats.pearsonr(m[c], m.y_conc)
    rs = stats.spearmanr(m[c], m.y_conc)
    print(f"{c:10s} pearson r={rp.statistic:+.3f} (p={rp.pvalue:.2e})  spearman ρ={rs.statistic:+.3f} (p={rs.pvalue:.2e})")

print("\n=== 组内相关(同一孕妇重复测量) ICC 粗略估计 ===")
# 组内 vs 组间方差 (one-way ANOVA)
grp = [d.y_conc.values for _, d in m.groupby("pid")]
F, p = stats.f_oneway(*[g for g in grp if len(g) > 1])
print(f"ANOVA F={F:.1f} p={p:.2e} -> 个体间差异显著, 需要个体水平建模")

print("\n=== 达标样本最早出现孕周(按 1 周窗口的达标率) ===")
m["wk0"] = np.floor(m.week)
tt = m.groupby("wk0").apply(lambda d: pd.Series({"n": len(d), "rate": (d.y_conc >= Y_TH).mean()}),
                            include_groups=False)
print(tt.round(3).to_string())
