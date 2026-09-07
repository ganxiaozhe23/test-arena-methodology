# -*- coding: utf-8 -*-
"""
问题1: 胎儿 Y 染色体浓度与孕周、BMI 等指标的相关特性分析 (v2)
- 相关分析 (行级 + 个体级 + 偏相关)
- 关系模型: 单/多变量 OLS -> 线性混合模型 (随机截距+随机斜率), 嵌套检验
- 达标概率 logit 模型
- 输出: 结果 JSON + 图
"""
import sys, os, json, warnings
sys.path.insert(0, "solution")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from common import load_male, Y_TH

fp = None
for cand in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    if os.path.exists(cand):
        fp = cand; break
if fp:
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
os.makedirs("figs", exist_ok=True)

m = load_male().dropna(subset=["y_conc", "week", "bmi", "age"]).copy()
m = m[m.week <= 28]
m["week_c"] = m.week - 11
m["bmi_c"] = m.bmi - 32.3
m["age_c"] = m.age - 29
m["pass01"] = (m.y_conc >= Y_TH).astype(int)
print(f"有效样本: {len(m)} 行 / {m.pid.nunique()} 孕妇, 孕周 {m.week.min():.1f}-{m.week.max():.1f}")
R = {}

# ================= 1) 相关分析 =================
rows = []
for c in ["week", "bmi", "age", "height", "weight", "raw_reads", "uniq_reads",
          "map_ratio", "dup_ratio", "gc", "flt_ratio"]:
    rp = stats.pearsonr(m[c], m.y_conc)
    rs = stats.spearmanr(m[c], m.y_conc)
    rows.append(dict(var=c, pearson=round(rp.statistic, 3), p_pearson=float(rp.pvalue),
                     spearman=round(rs.statistic, 3), p_spearman=float(rs.pvalue)))
print("== 行级相关 ==")
print(pd.DataFrame(rows).round(4).to_string(index=False))

per = m.groupby("pid").agg(y=("y_conc", "mean"), week=("week", "mean"), bmi=("bmi", "first"),
                           age=("age", "first"), height=("height", "first"), weight=("weight", "first"))
rows_p = []
for c in ["bmi", "age", "weight", "height"]:
    rs = stats.spearmanr(per[c], per.y)
    rp = stats.pearsonr(per[c], per.y)
    rows_p.append(dict(var=c, pearson=round(rp.statistic, 3), p_pearson=float(rp.pvalue),
                       spearman=round(rs.statistic, 3), p_spearman=float(rs.pvalue)))
print("\n== 个体级相关 (n=%d) ==" % len(per))
print(pd.DataFrame(rows_p).round(4).to_string(index=False))
R["corr_row"] = rows; R["corr_person"] = rows_p

# ================= 2) 关系模型 =================
def fit_lmm(formula, re="~ week_c", reml=True):
    mm = smf.mixedlm(formula, data=m, groups=m["pid"], re_formula=re).fit(reml=reml)
    return mm

res_models = {}
# --- AIC 对照 (全部 ML) ---
o0 = smf.ols("y_conc ~ week_c", data=m).fit()
o1 = smf.ols("y_conc ~ week_c + bmi_c", data=m).fit()
l1 = fit_lmm("y_conc ~ week_c", re="~1", reml=False)
l2 = fit_lmm("y_conc ~ week_c + bmi_c", re="~1", reml=False)
l3 = fit_lmm("y_conc ~ week_c + bmi_c", re="~ week_c", reml=False)
for name, mm in [("OLS_week", o0), ("OLS_week+bmi", o1), ("LMM_ri_week", l1),
                 ("LMM_ri_week+bmi", l2), ("LMM_ris_week+bmi", l3)]:
    res_models[name] = dict(aic=float(mm.aic), bic=float(mm.bic), ll=float(mm.llf))
    print(f"{name:22s} AIC={mm.aic:9.2f} BIC={mm.bic:9.2f}")

# 嵌套检验
LR_ri = 2*(l2.llf - l1.llf); p_ri = stats.chi2.sf(LR_ri, 1)
LR_rs = 2*(l3.llf - l2.llf); p_rs = stats.chi2.sf(LR_rs, 2)
print(f"\n随机截距检验: LR={LR_ri:.2f} p={p_ri:.2e};  随机斜率检验: LR={LR_rs:.2f} p={p_rs:.2e}")

# 主模型: 最终 LMM (REML)
main = fit_lmm("y_conc ~ week_c + bmi_c", re="~ week_c", reml=True)
fe = pd.DataFrame(dict(coef=main.params.iloc[:3], se=main.bse.iloc[:3],
                       t=main.tvalues.iloc[:3], p=main.pvalues.iloc[:3]))
print("\n== 最终模型固定效应 (LMM: 随机截距+随机斜率) ==")
print(fe.round(4).to_string())
R["final_FE"] = fe.round(4).to_dict("index")
R["cov_re"] = main.cov_re.round(5).to_dict()
R["scale"] = float(main.scale)

vc = main.cov_re
v_u, v_s, cov_us = vc.iloc[0, 0], vc.iloc[1, 1], vc.iloc[0, 1]
icc = v_u / (v_u + main.scale)
# 边际/条件 R²
exog_f = main.model.exog
params_f = main.params.iloc[:exog_f.shape[1]]
fixed = exog_f @ params_f
var_f = np.var(fixed)
R2m = var_f / (var_f + v_u + v_s + main.scale)
R2c = (var_f + v_u + v_s) / (var_f + v_u + v_s + main.scale)
print(f"\n随机效应: σ²_u={v_u:.5f} (个体水平方差), σ²_s={v_s:.6f} (斜率方差), "
      f"σ²_e={main.scale:.6f}\nICC={icc:.3f}  边际R²={R2m:.3f}  条件R²={R2c:.3f}")
R["icc"] = icc; R["R2m"] = R2m; R["R2c"] = R2c

# 稳健性: 加年龄/其他协变量的 LMM 固定效应 (检验 BMI 稳健)
rob = fit_lmm("y_conc ~ week_c + bmi_c + age_c", re="~ week_c", reml=True)
print("\n== 稳健性模型 (+年龄) 固定效应 ==")
print(pd.DataFrame(dict(coef=rob.params.iloc[:4], p=rob.pvalues.iloc[:4])).round(4).to_string())
R["robust_age"] = pd.DataFrame(dict(coef=rob.params.iloc[:4], p=rob.pvalues.iloc[:4])).round(4).to_dict("index")

# 残差诊断
rs_std = main.resid / np.sqrt(main.scale)
print(f"\n残差Shapiro W={stats.shapiro(rs_std[:5000]).statistic:.3f} (p≈0 因n大; 看QQ图直观看尾部)")
print(f"残差均值={rs_std.mean():.3f}, 标准差={rs_std.std():.3f}")

# BLUP
bl = pd.DataFrame([(k, v["Group"], v["week_c"]) for k, v in main.random_effects.items()],
                  columns=["pid", "u0", "u1"])
bl = bl.merge(per.reset_index()[["pid", "bmi", "age", "weight", "height"]], on="pid")
R["blup_u0_range"] = [float(bl.u0.min()), float(bl.u0.quantile(.05)), float(bl.u0.quantile(.5)),
                      float(bl.u0.quantile(.95)), float(bl.u0.max())]
print("\nBLUP u0 分位(min,p5,中位,p95,max):", np.round(R["blup_u0_range"], 4))
bl.to_csv("solution/blup_random_effects.csv", index=False)

# ================= 3) 达标概率 logit =================
lg = smf.logit("pass01 ~ week_c + bmi_c + age_c", data=m).fit(disp=0)
print("\n== 达标概率 logit ==")
print(pd.DataFrame(dict(coef=lg.params, p=lg.pvalues)).round(4).to_string())
R["logit"] = dict(coef=lg.params.round(4).to_dict(), p=lg.pvalues.round(4).to_dict())
wgrid = np.arange(10, 26.01, 0.5)
prob_pred = {f"BMI{int(b)}": np.round(1/(1+np.exp(-(lg.params[0]+lg.params[1]*(wgrid-11)
                + lg.params[2]*(b-32.3) + lg.params[3]*0))), 4).tolist() for b in [24, 28, 32, 36, 40, 44]}
R["prob_week_bmi"] = dict(weeks=wgrid.tolist(), prob=prob_pred)

# ================= 图 =================
# 图5: logit 达标概率曲线
fig, ax = plt.subplots(figsize=(8, 4.6))
for b in [24, 28, 32, 36, 40, 44]:
    ph = 1/(1+np.exp(-(lg.params[0] + lg.params[1]*(wgrid-11) + lg.params[2]*(b-32.3))))
    ax.plot(wgrid, ph, lw=2, label=f"BMI={b}")
ax.axhline(.9, color="gray", ls=":")
ax.axvline(12, color="r", ls="--", lw=.8, alpha=.6)
ax.set_xlabel("孕周"); ax.set_ylabel("达标概率 P(Y≥4%)")
ax.set_title("不同 BMI 下 Y 浓度达标概率随孕周变化 (logit)")
ax.legend(ncol=2, fontsize=9)
plt.tight_layout(); plt.savefig("figs/fig5_passprob.png"); plt.close()

# 图6: 模型预测曲线 + 观测散点
fig = plt.figure(figsize=(12.5, 4.8))
ax = fig.add_subplot(121)
for b in [24, 28, 32, 36, 40, 44]:
    pred = (main.params[0] + main.params[2]*(b-32.3)) + main.params[1]*(wgrid-11)
    ax.plot(wgrid, pred, lw=1.8, label=f"BMI={b}")
ax.axhline(Y_TH, color="r", ls="--", lw=1)
ax.set_xlabel("孕周"); ax.set_ylabel("预测 Y 浓度 (固定效应)")
ax.set_title("(a) LMM 固定效应: 浓度-孕周曲线 (BMI 分层)")
ax.legend(fontsize=8, ncol=2)
ax2 = fig.add_subplot(122)
sc = ax2.scatter(m.week, m.y_conc, c=m.bmi, s=8, alpha=.5, cmap="viridis")
ax2.axhline(Y_TH, color="r", ls="--", lw=1)
ax2.set_xlabel("孕周"); ax2.set_ylabel("Y 浓度"); ax2.set_title("(b) 观测散点")
plt.colorbar(sc, ax=ax2, label="BMI")
plt.tight_layout(); plt.savefig("figs/fig6_surface.png"); plt.close()

# 图7: 个体随机效应与 BMI
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
axes[0].scatter(bl.bmi, bl.u0*100, s=14, alpha=.6)
z = np.polyfit(bl.bmi, bl.u0*100, 1)
xx = np.linspace(bl.bmi.min(), bl.bmi.max(), 50)
axes[0].plot(xx, np.polyval(z, xx), "r-", lw=1.8)
axes[0].set_xlabel("BMI"); axes[0].set_ylabel("个体随机截距 u0 (百分点)")
axes[0].set_title("(a) 个体水平差异 vs BMI")
axes[1].scatter(bl.bmi, bl.u1*1000, s=14, alpha=.6, color="#C44E52")
axes[1].axhline(0, color="gray", lw=.8)
axes[1].set_xlabel("BMI"); axes[1].set_ylabel("随机斜率 u1 (×1000/周)")
axes[1].set_title("(b) 个体增长斜率 vs BMI")
plt.tight_layout(); plt.savefig("figs/fig7_blup.png"); plt.close()

# 图8: 达标时间轮廓: 等达标率曲线 (week × bmi)
bgrid = np.arange(24, 47, 1)
surf = np.zeros((len(wgrid), len(bgrid)))
for i, w in enumerate(wgrid):
    for j, b in enumerate(bgrid):
        surf[i, j] = 1/(1+np.exp(-(lg.params[0] + lg.params[1]*(w-11) + lg.params[2]*(b-32.3))))
fig, ax = plt.subplots(figsize=(8, 4.6))
CS = ax.contour(bgrid, wgrid, surf, levels=[0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98], cmap="viridis")
ax.clabel(CS, inline=True, fontsize=8, fmt="%.2f")
ax.set_xlabel("BMI"); ax.set_ylabel("孕周")
ax.set_title("Y 浓度达标概率等值线 (logit 模型)")
plt.tight_layout(); plt.savefig("figs/fig8_contour.png"); plt.close()

with open("solution/q1_results.json", "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1, default=float)
print("\nsaved solution/q1_results.json + figs/fig5-8")
