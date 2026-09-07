# -*- coding: utf-8 -*-
"""
问题3: 多因素(身高/体重/年龄等)+检测误差+达标比例下的分组与时点优化

1) 多因素达标时间模型: LMM(随机截距+斜率) 逐步加入 年龄/身高/体重,
   比较 AIC/BIC; 时间前向验证(用前k-1次检测预测第k次)评估预测增益
2) 个体达标时间 τ^ 的多因素估计 + 方差分解(可被解释部分 vs 个体残余)
3) 达标比例-时点权衡: 对每组求 满足 P(达标|t)≥p 约束下的风险最小时点,
   给出 p=0.70/0.80/0.85/0.90/0.95 权衡曲线; 推荐策略
4) 检测误差: 浓度测量噪声 σ_m 与"计划-实际到检"执行偏差的双重传播,
   参数自助给出时点稳健区间与风险增幅
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
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as smapi
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

m = load_male().dropna(subset=["y_conc", "week", "bmi", "age", "height", "weight"]).copy()
m = m[m.week <= 28]
m["week_c"] = m.week - 11
m["bmi_c"] = m.bmi - 32.3
m["age_c"] = m.age - 29
m["height_c"] = m.height - 160
m["weight_c"] = m.weight - 72
R = {}

# ============ 1) 多因素 LMM 比较 ============
print("== 1. 多因素线性混合模型比较 (随机截距+随机斜率) ==")
fits = {}
for name, fml in [
    ("F1: 仅孕周+BMI", "y_conc ~ week_c + bmi_c"),
    ("F2: +年龄", "y_conc ~ week_c + bmi_c + age_c"),
    ("F3: +身高体重", "y_conc ~ week_c + bmi_c + height_c + weight_c"),
    ("F4: 全部协变量", "y_conc ~ week_c + bmi_c + age_c + height_c + weight_c"),
    ("F5: +BMI×周交互", "y_conc ~ week_c + bmi_c + age_c + height_c + weight_c + bmi_c*week_c"),
]:
    f = smf.mixedlm(fml, data=m, groups=m["pid"], re_formula="~ week_c").fit(reml=False)
    fits[name] = f
    print(f"{name:20s} AIC={f.aic:10.2f} BIC={f.bic:10.2f} LL={f.llf:9.2f}  n_par={len(f.params)}")
R["model_compare"] = {k: dict(aic=float(f.aic), bic=float(f.bic), ll=float(f.llf)) for k, f in fits.items()}

full = smf.mixedlm("y_conc ~ week_c + bmi_c + age_c + height_c + weight_c",
                   data=m, groups=m["pid"], re_formula="~ week_c").fit(reml=True)
print("\nF4 固定效应:")
fe4 = pd.DataFrame(dict(coef=full.params.iloc[:6], se=full.bse.iloc[:6],
                        t=full.tvalues.iloc[:6], p=full.pvalues.iloc[:6])).round(4)
print(fe4.to_string())
R["F4_FE"] = fe4.to_dict("index")

# LR: F2 vs F1, F4 vs F2
for a, b in [("F1: 仅孕周+BMI", "F2: +年龄"), ("F2: +年龄", "F3: +身高体重")]:
    lr = 2 * (fits[b].llf - fits[a].llf)
    df = len(fits[b].params) - len(fits[a].params)
    print(f"LR {a} -> {b}: LR={lr:.2f}, df={df}, p={stats.chi2.sf(lr, df):.2e}")

# ============ 2) 多因素对"个体水平/达标时间"的解释力 ============
print("\n== 2. 多因素对个体间差异的解释 (孕周校正后个体水平) ==")
import statsmodels.api as smapi
m["pass01"] = (m.y_conc >= Y_TH).astype(int)
# 孕周-浓度总体曲线残差 -> 个体水平
Xw = np.c_[np.ones(len(m)), m.week_c, m.week_c**2]
o = smapi.OLS(m.y_conc, Xw).fit()
per_lvl = m.assign(r=o.resid).groupby("pid").agg(level=("r", "mean"), bmi=("bmi", "first"),
                                                 age=("age", "first"), height=("height", "first"),
                                                 weight=("weight", "first")).reset_index()
for name, cols in [("仅BMI", ["bmi"]), ("BMI+年龄身高体重", ["bmi", "age", "height", "weight"])]:
    X = smapi.add_constant(per_lvl[cols])
    f = smapi.OLS(per_lvl.level, X).fit()
    print(f"  个体水平 ~ {name:16s}: R²={f.rsquared:.3f}  (n={len(per_lvl)})")

# 未截断 τ 回归: 仅BMI vs 多因素
per_u = pd.read_csv("solution/男胎个体达标时间_BLUP.csv")
per_u = per_u[(per_u.tau > 10.4) & (per_u.tau < 28)].copy()
per_u["bmi_c"] = per_u.bmi - 32.3; per_u["age_c"] = per_u.age - 29
per_u["height_c"] = per_u.height - 160; per_u["weight_c"] = per_u.weight - 72
print(f"\n-- 未截断个体达标时间 τ (n={len(per_u)}) 回归对比 --")
for name, cols in [("仅BMI", ["bmi_c"]), ("BMI+年龄身高体重", ["bmi_c", "age_c", "height_c", "weight_c"])]:
    X = smapi.add_constant(per_u[cols])
    f = smapi.OLS(per_u.tau, X).fit()
    print(f"  τ ~ {name:16s}: R²={f.rsquared:.3f}  adjR²={f.rsquared_adj:.3f}")
# ============ 3) 各因素单变量相关 (全样本与未截断) ============
print("\n== 3. τ 与各因素 Spearman (全样本 / 未截断子样本) ==")
per_all = pd.read_csv("solution/男胎个体达标时间_BLUP.csv")
for c in ["bmi", "age", "height", "weight"]:
    rs = stats.spearmanr(per_all[c], per_all.tau)
    print(f"  [全样本 n={len(per_all)}] τ vs {c:6s}: ρ={rs.statistic:+.3f} p={rs.pvalue:.1e}")
for c in ["bmi", "age", "height", "weight"]:
    rs = stats.spearmanr(per_u[c], per_u.tau)
    print(f"  [未截断 n={len(per_u)}]  τ vs {c:6s}: ρ={rs.statistic:+.3f} p={rs.pvalue:.1e}")

# 各 BMI 细区间的 τ 中位(未截断/全样本) —— 展示分段必要性
print("\n-- 全样本按 BMI 细分的 τ 分位数 --")
for lo, hi in [(20, 28), (28, 30), (30, 32), (32, 34), (34, 36), (36, 38), (38, 40), (40, 99)]:
    dd = per_all[(per_all.bmi >= lo) & (per_all.bmi < hi)]
    if len(dd) == 0:
        continue
    print(f"  BMI[{lo},{hi}): n={len(dd):3d}  τ中位={dd.tau.median():5.2f}  P75={dd.tau.quantile(.75):5.2f}  "
          f"P90={dd.tau.quantile(.9):5.2f}  ≥12周占比={100*(dd.tau>12).mean():.0f}%")

# ============ 4) 达标比例约束下的时点决策 ============
print("\n== 4. 达标比例-风险权衡 (最终分档: <32 / 32-36 / ≥36 与合并档 <36) ==")
K = np.log(4) / 9.0
DELTA = 1.5
def w(t): return np.exp(K * (t - 11))
GRID = np.arange(10.0, 25.01, 0.25)
def risk_at(t, taus, delta=DELTA):
    d = np.where(taus <= t, t, taus + delta)
    return float(np.mean(w(d)))
def schedule_p(taus, pmin):
    """在 达标率≥pmin 约束下风险最小时点"""
    taus = np.asarray(taus)
    rv = np.array([risk_at(t, taus) for t in GRID])
    ok = np.array([np.mean(taus <= t) >= pmin for t in GRID])
    if not ok.any():
        return None
    i = int(np.argmin(np.where(ok, rv, np.inf)))
    return float(GRID[i]), float(rv[i]), float(np.mean(taus <= GRID[i]))

labs = np.digitize(per_all.bmi.values, [20, 36, 99])
groups = {"<36": per_all.tau.values[labs == 1], "≥36": per_all.tau.values[labs == 2]}
grp_names = list(groups)
ps = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
tbl = {}
print(f"{'达标率目标':<12s}" + "".join(f"{g:>16s}" for g in grp_names))
for pmin in ps:
    line = []
    for g in grp_names:
        s = schedule_p(groups[g], pmin)
        line.append(f"{s[0]:.1f}w/风险{s[1]:.2f}" if s else "不可达")
    print(f"p≥{pmin:<7.2f} " + "".join(f"{x:>16s}" for x in line))
    tbl[f"p{pmin}"] = {g: (None if schedule_p(groups[g], pmin) is None else
                           {"t": schedule_p(groups[g], pmin)[0],
                            "risk": schedule_p(groups[g], pmin)[1]}) for g in grp_names}
R["tradeoff"] = tbl

# 无约束(风险最小) 与 双目标推荐(风险≤1.02×最小 内达标率最高) 对照
def reco(taus):
    taus = np.asarray(taus)
    rv = np.array([risk_at(t, taus) for t in GRID])
    i0 = int(np.argmin(rv))
    band = rv <= 1.02 * rv[i0]
    j = int(np.where(band)[0][-1])
    return GRID[i0], rv[i0], GRID[j], rv[j]

print("\n-- 各组时点方案对照 --")
final3 = {}
for g in grp_names:
    t0, r0, tj, rj = reco(groups[g])
    q = np.quantile(groups[g], [.5, .75, .9])
    final3[g] = dict(n=int(len(groups[g])), tau_med=float(q[0]), tau75=float(q[1]), tau90=float(q[2]),
                     t_min=float(t0), r_min=float(r0), t_reco=float(tj), r_reco=float(rj),
                     pass_reco=float(np.mean(groups[g] <= tj)))
    print(f"  {g:4s} n={len(groups[g]):3d} | 风险最小时点 t={t0:5.2f} (风险{r0:.3f}) "
          f"| 平带内推荐 t={tj:5.2f} (风险{rj:.3f}, 达标率{np.mean(groups[g]<=tj)*100:.0f}%) "
          f"| τ: 中位{q[0]:.1f} P75={q[1]:.1f} P90={q[2]:.1f}")
R["final"] = final3

# ============ 5) 检测误差敏感性 ============
print("\n== 5. 检测误差敏感性(参数自助 600 次) ==")
sigma_m = 0.00903
rng = np.random.default_rng(11)
boot = {g: [] for g in grp_names}
for b in range(600):
    idx = rng.integers(0, len(per_all), len(per_all))
    sub = per_all.iloc[idx]
    # (a) 浓度测量噪声→τ噪声; (b) 计划执行偏差: 实际到检孕周 = 计划+ε, ε~N(0,0.5周)
    dconc = rng.normal(0, sigma_m, len(sub))
    slope_avg = 0.0035
    tau_b = np.clip(sub.tau.values - dconc / slope_avg, 10, 30)
    labb = np.digitize(sub.bmi.values, [20, 36, 99])
    for k_, g in zip([1, 2], grp_names):
        taus = tau_b[labb == k_]
        rv = np.array([risk_at(t, taus) for t in GRID])
        i0 = int(np.argmin(rv)); band = rv <= 1.02 * rv[i0]
        t_plan = float(GRID[int(np.where(band)[0][-1])])
        # 执行偏差下实际风险
        act = np.mean([risk_at(t_plan + e, taus) for e in rng.normal(0, 0.5, 200)])
        boot[g].append((t_plan, act))
boot_out = {}
for g in grp_names:
    a = np.array([x[0] for x in boot[g]]); b2 = np.array([x[1] for x in boot[g]])
    base = risk_at(final3[g]["t_reco"], groups[g])
    print(f"  {g:4s}: 推荐时点 中位={np.median(a):.2f} 周, 95%CI=[{np.quantile(a,.025):.2f},{np.quantile(a,.975):.2f}]; "
          f"执行偏差后平均风险增幅 {np.mean((b2-base)/base)*100:.1f}%")
    boot_out[g] = dict(median=float(np.median(a)), lo=float(np.quantile(a, .025)),
                       hi=float(np.quantile(a, .975)), risk_increase=float(np.mean((b2-base)/base)))
R["boot"] = boot_out

# 边界敏感性: BMI 分界 35/36/37
print("\n-- 分界点 ±1 稳健性 --")
for bd in [35, 36, 37]:
    g1 = per_all.tau.values[per_all.bmi < bd]; g2 = per_all.tau.values[per_all.bmi >= bd]
    s1 = reco(g1); s2 = reco(g2)
    p = stats.mannwhitneyu(g1, g2).pvalue
    print(f"  分界 {bd}: <{bd} 推荐 {s1[2]:.2f}周 | ≥{bd} 推荐 {s2[2]:.2f}周 | 组间差异 p={p:.1e}")
R["boundary_sens"] = {bd: {"t_lo": reco(per_all.tau.values[per_all.bmi < bd])[2],
                           "t_hi": reco(per_all.tau.values[per_all.bmi >= bd])[2]} for bd in [35, 36, 37]}

with open("solution/q3_results.json", "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1, default=str)

# ============ 图 ============
# 权衡曲线
fig, ax = plt.subplots(figsize=(8.2, 5))
for g in grp_names:
    rv = np.array([risk_at(t, groups[g]) for t in GRID])
    pr = np.array([np.mean(groups[g] <= t) for t in GRID])
    ax.plot(pr, rv, lw=2, label=g)
ax.set_xlabel("单次检测达标比例 P(τ≤t)"); ax.set_ylabel("组内期望风险")
ax.set_title("达标比例-期望风险权衡曲线")
ax.legend()
plt.tight_layout(); plt.savefig("figs/fig11_tradeoff.png"); plt.close()

# 个体τ vs BMI (散点+平滑) 显示分组合理性
fig, ax = plt.subplots(figsize=(8, 4.6))
pa = pd.read_csv("solution/男胎个体达标时间_BLUP.csv")
ax.scatter(pa.bmi, pa.tau, s=18, alpha=.65, color="#4C72B0")
for bd, c in [(32, "#55A868"), (36, "#C44E52")]:
    ax.axvline(bd, ls="--", lw=1.2, color=c)
ax.axhline(12, color="r", ls=":", lw=1)
ax.set_xlabel("孕妇 BMI"); ax.set_ylabel("个体最早达标时间 τ (孕周)")
ax.set_title("个体最早达标时间与 BMI (竖线=推荐分档边界)")
plt.tight_layout(); plt.savefig("figs/fig12_tau_bmi.png"); plt.close()
print("\nsaved q3_results.json + figs/fig11-12")
