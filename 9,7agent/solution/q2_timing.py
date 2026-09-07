# -*- coding: utf-8 -*-
"""
问题2 (v2): 按 BMI 合理分组 + 各组最佳 NIPT 时点
- 测量噪声估计(同周重复检测)、个体最早达标时间 τ(BLUP法, 插值法对照)
- 风险函数 w(发现孕周): 平滑 1:4:16 (11周=1, 20周=4, 29周=16)
- 发现孕周 d_i(t) = t (τ≤t 一次检测可靠) 或 τ+Δ (未达标需复查)
- 判据: 风险平带(≤2%超最小风险)内取"单次达标率最高"时点(临床双目标)
- 分组: 数据驱动联合优化分界点(k=1,2) vs 经验分档对照
- 检测误差影响: 参数自助法 -> 时点不确定区间
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
R = {}

# ---------- 0) 测量噪声 ----------
dup = m.groupby(["pid", "week"]).filter(lambda g: len(g) >= 2)
g = dup.groupby(["pid", "week"])
ss = g.apply(lambda d: np.var(d.y_conc, ddof=1) * (len(d) - 1), include_groups=False).sum()
sg = g.size().sum() - g.ngroups
sigma_m = float(np.sqrt(ss / sg))
print(f"同日重复检测测量噪声 σ_m = {sigma_m:.5f} (共 {g.ngroups} 对)")
R["sigma_meas"] = sigma_m

# ---------- 1) BLUP 法个体 τ ----------
import statsmodels.formula.api as smf
mm = smf.mixedlm("y_conc ~ week_c + bmi_c", data=m, groups=m["pid"],
                 re_formula="~ week_c").fit(reml=True)
b0, bw, bb = mm.params["Intercept"], mm.params["week_c"], mm.params["bmi_c"]
rows = []
for pid, v in mm.random_effects.items():
    u0, u1 = v["Group"], v["week_c"]
    sub = m[m.pid == pid]
    tau = np.clip(11 + (Y_TH - (b0 + u0 + bb * (sub.bmi.iloc[0] - 32.3))) / (bw + u1), 10, 30)
    rows.append(dict(pid=pid, bmi=sub.bmi.iloc[0], age=sub.age.iloc[0],
                     height=sub.height.iloc[0], weight=sub.weight.iloc[0],
                     u0=float(u0), u1=float(u1), tau=float(tau)))
per = pd.DataFrame(rows)
iv = pd.read_csv("solution/男胎个体达标时间_插值法.csv").rename(columns={"t": "tau_iv"})
cmpd = per.merge(iv[["pid", "tau_iv"]], on="pid", how="left").dropna(subset=["tau_iv"])
rho = stats.spearmanr(cmpd.tau, cmpd.tau_iv)
print(f"BLUP τ vs 插值 τ: ρ={rho.statistic:.3f} (n={len(cmpd)}), 中位差={np.median(cmpd.tau-cmpd.tau_iv):+.2f}w")
R["tau_corr"] = float(rho.statistic)
per.to_csv("solution/男胎个体达标时间_BLUP.csv", index=False)

# ---------- 2) 风险/判据 ----------
K = np.log(4) / 9.0
DELTA = 1.5          # 未达标复查: 复查安排在最早达标后约1.5周(敏感性 0.5~3)
EPS = 1.02           # 风险平带容差
def w(t): return np.exp(K * (t - 11))
def risk_at(t, taus, delta=DELTA):
    taus = np.asarray(taus)
    d = np.where(taus <= t, t, taus + delta)
    return float(np.mean(w(d)))
GRID = np.arange(10.0, 25.01, 0.25)

def group_schedule(taus, delta=DELTA, eps=EPS):
    """返回 (风险最小时点, 平带推荐时点, 达标率曲线等)"""
    taus = np.asarray(taus)
    rv = np.array([risk_at(t, taus, delta) for t in GRID])
    i_min = int(np.argmin(rv)); r_min = rv[i_min]
    band = rv <= eps * r_min
    # 平带内取达标率最高(最晚)点 -> 单次可靠度最高
    cand = GRID[band]
    t_knee = cand[-1]
    j = int(np.argmin(np.abs(GRID - t_knee)))
    return dict(t_min=float(GRID[i_min]), r_min=float(r_min),
                t_reco=float(t_knee), r_reco=float(rv[j]),
                pass_reco=float(np.mean(taus <= t_knee)),
                pass_min=float(np.mean(taus <= GRID[i_min])), rv=rv)

def pass_rate(t, taus): return float(np.mean(np.asarray(taus) <= t))

print(f"风险函数: w(10)={w(10):.3f} w(11)={w(11):.2f} w(12)={w(12):.2f} w(20)={w(20):.2f} w(25)={w(25):.2f}")

# ---------- 3) 候选分档对照 ----------
def eval_part(sel_groups, nmin=10):
    """sel_groups: dict name->bool array"""
    out = {}
    for gn, sel in sel_groups.items():
        n = int(sel.sum())
        if n < nmin:
            out[gn] = dict(n=n, skip=True)
            continue
        s = group_schedule(per.tau.values[sel])
        out[gn] = dict(n=n, t_min=round(s["t_min"], 2), r_min=round(s["r_min"], 4),
                       t_reco=round(s["t_reco"], 2), r_reco=round(s["r_reco"], 4),
                       pass_reco=round(s["pass_reco"], 3), pass_min=round(s["pass_min"], 3))
    return out

sel_all = pd.Series(True, index=per.index)
cands = {
    "统一(无分组)": {"全体": sel_all},
    "P5_题示5档": {f"[{a},{b})" if b < 99 else f"[{a},∞)":
                  ((per.bmi >= a) & (per.bmi < b)).values
                  for a, b in [(20, 28), (28, 32), (32, 36), (36, 40), (40, 99)]},
    "P4a": {f"[{a},{b})" if b < 99 else f"[{a},∞)":
            ((per.bmi >= a) & (per.bmi < b)).values for a, b in [(20, 32), (32, 36), (36, 40), (40, 99)]},
    "P4b": {f"[{a},{b})" if b < 99 else f"[{a},∞)":
            ((per.bmi >= a) & (per.bmi < b)).values for a, b in [(20, 28), (28, 32), (32, 36), (36, 99)]},
    "P3": {f"[{a},{b})" if b < 99 else f"[{a},∞)":
           ((per.bmi >= a) & (per.bmi < b)).values for a, b in [(20, 32), (32, 36), (36, 99)]},
}
print("\n===== 各分档方案: 每组最佳时点(风险最小 / 平带推荐) =====")
plan = {}
for name, groups in cands.items():
    res = eval_part(groups)
    plan[name] = res
    tot_min = tot_reco = tot_pass = 0
    parts = []
    for gn, d in res.items():
        if d.get("skip"):
            parts.append(f"  {gn:10s} n={d['n']:3d} (样本不足跳过)"); continue
        parts.append(f"  {gn:10s} n={d['n']:3d} | 风险最小时点 {d['t_min']:5.2f}周(风险{d['r_min']:.3f}) "
                     f"| 平带推荐 {d['t_reco']:5.2f}周(风险{d['r_reco']:.3f}, 达标率{d['pass_reco']*100:.0f}%)")
        tot_min += d["r_min"] * d["n"]; tot_reco += d["r_reco"] * d["n"]; tot_pass += d["pass_reco"] * d["n"]
    print(f"--- {name} (平均风险: min基准 {tot_min/len(per):.4f} | 推荐时点 {tot_reco/len(per):.4f}, 平均达标率 {tot_pass/len(per)*100:.0f}%)")
    for p in parts: print(p)
    plan[name]["_avg"] = dict(risk_min=round(tot_min/len(per), 4), risk_reco=round(tot_reco/len(per), 4))
R["plans"] = {k: {kk: {a: bb for a, bb in vv.items() if a != "rv"} for kk, vv in v.items()}
              for k, v in plan.items()}

# ---------- 4) 数据驱动分界点联合优化 (k=1,2) ----------
print("\n===== 数据驱动分界点优化 =====")
bmis = np.sort(per.bmi.values)
candidates = np.unique(np.round(np.arange(28, 46.5, 0.5), 1))

def total_risk(bounds):
    """bounds: 升序分界点列表; 分组 [min,b1),[b1,b2)..."""
    lab = np.zeros(len(per), dtype=int)
    edges = [20] + list(bounds) + [99]
    for i in range(len(edges) - 1):
        lab[(per.bmi >= edges[i]) & (per.bmi < edges[i+1])] = i
    tot = 0.0
    for k in np.unique(lab):
        sel = lab == k
        if sel.sum() < 10:
            return np.inf
        s = group_schedule(per.tau.values[sel])
        tot += s["r_reco"] * sel.sum()     # 用推荐时点风险
    return tot / len(per)

best = {}
for k in [1, 2]:
    if k == 1:
        cand = [[b] for b in candidates if (per.bmi < b).sum() >= 15 and (per.bmi >= b).sum() >= 10]
    else:
        cand = [[b1, b2] for b1 in candidates for b2 in candidates if b2 > b1 + 1
                and (per.bmi < b1).sum() >= 15 and ((per.bmi >= b1) & (per.bmi < b2)).sum() >= 10
                and (per.bmi >= b2).sum() >= 10]
    tr = [(b, total_risk(b)) for b in cand]
    tr = [(b, r) for b, r in tr if np.isfinite(r)]
    tr.sort(key=lambda x: x[1])
    best[k] = tr[:5]
    for b, r in tr[:5]:
        print(f"k={k} 分界 {b} -> 平均推荐风险 {r:.4f}")
R["boundary_search"] = {str(k): [dict(b=b, risk=float(r)) for b, r in best[k][:3]] for k in best}

# ---------- 5) 推荐方案细表 + 组间检验 ----------
rec_edges = [20, 32, 36, 99]
labs = np.digitize(per.bmi, rec_edges)  # 1:[20,32) 2:[32,36) 3:[36,99)
print("\n===== 推荐 3 档方案 [20,32) [32,36) [36,∞) =====")
final = {}
for k in sorted(set(labs)):
    sel = labs == k
    s = group_schedule(per.tau.values[sel])
    lo, hi = rec_edges[k-1], rec_edges[k]
    final[f"[{lo},{hi})" if hi < 99 else f"[{lo},∞)"] = dict(
        n=int(sel.sum()), lo=lo, hi=hi,
        t_med=float(np.median(per.tau.values[sel])), t_q75=float(np.quantile(per.tau.values[sel], .75)),
        t_reco=s["t_reco"], t_min=s["t_min"], risk_reco=round(s["r_reco"], 4),
        pass_reco=round(s["pass_reco"], 3))
    print(f"  [{lo},{hi})" if hi < 99 else f"  [{lo},∞)", end="")
    print(f"  n={int(sel.sum()):3d}  τ中位={np.median(per.tau.values[sel]):4.1f} τ75%={np.quantile(per.tau.values[sel],.75):4.1f}  "
          f"推荐时点={s['t_reco']:.2f}周 (风险={s['r_reco']:.3f}, 单次达标率={s['pass_reco']*100:.0f}%)  "
          f"[风险最小={s['t_min']:.2f}, 风险={s['r_min']:.3f}]")
R["final"] = final

# Kruskal-Wallis: 三组 τ 差异
sel1, sel2, sel3 = [labs == k for k in [1, 2, 3]]
kw = stats.kruskal(per.tau.values[sel1], per.tau.values[sel2], per.tau.values[sel3])
print(f"\nKruskal-Wallis 组间 τ 差异: H={kw.statistic:.2f}, p={kw.pvalue:.2e}")
print("两两 Mann-Whitney: 1v2 p=%.2e, 2v3 p=%.2e, 1v3 p=%.2e" % tuple(
    stats.mannwhitneyu(per.tau.values[a], per.tau.values[b]).pvalue
    for a, b in [(sel1, sel2), (sel2, sel3), (sel1, sel3)]))
# BMI 落在边界附近的稳健性: 边界 ±1 BMI 时 τ 差异是否仍显著
for bd, (a, b) in {"36边界": (per.bmi < 36, per.bmi >= 36), "32边界": (per.bmi < 32, per.bmi >= 32)}.items():
    if a.sum() >= 15 and b.sum() >= 15:
        p = stats.mannwhitneyu(per.tau.values[a], per.tau.values[b]).pvalue
        print(f"  边界稳健性 {bd}: Mann-Whitney p={p:.2e}")

# ---------- 6) 检测误差敏感性(自助) ----------
rng = np.random.default_rng(7)
print("\n===== 检测误差/参数敏感性 (参数自助 500 次) =====")
boot = {gname: [] for gname in final}
deltas = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
for b in range(500):
    idx = rng.integers(0, len(per), len(per))
    sub = per.iloc[idx]
    # 浓度测量噪声 → τ 噪声 (斜率 1/(bw+u1) ≈ 286 周/单位浓度)
    dconc = rng.normal(0, sigma_m, len(sub))
    tau_b = np.clip(sub.tau.values - dconc / np.clip(bw + sub.u1.values, 1e-4, None), 10, 30)
    delta = float(rng.choice(deltas))
    Kb = K * rng.uniform(0.85, 1.15)
    labb = np.digitize(sub.bmi.values, rec_edges)
    for k in sorted(set(labb)):
        sel = labb == k
        taus = tau_b[sel]
        rv = np.array([np.mean(np.exp(Kb * (np.where(taus <= t, t, taus + delta) - 11))) for t in GRID])
        r_min = rv.min()
        band = rv <= 1.02 * r_min
        j = int(np.where(band)[0][-1])
        boot[list(final.keys())[k-1]].append(float(GRID[j]))
sens_out = {}
for gn in final:
    a = np.array(boot[gn])
    sens_out[gn] = dict(median=float(np.median(a)), q2_5=float(np.quantile(a, .025)),
                        q97_5=float(np.quantile(a, .975)))
    print(f"  {gn:10s}: 推荐时点中位 {np.median(a):5.2f} 周, 95%区间 [{np.quantile(a,.025):.2f}, {np.quantile(a,.975):.2f}]")
R["sensitivity_boot"] = sens_out

# Δ 单项敏感性
print("\n-- 复查延迟 Δ 单项敏感性 (推荐时点, 其余固定) --")
for gn, k_ in zip(final, [1, 2, 3]):
    sel = labs == k_
    line = []
    for dlt in deltas:
        s = group_schedule(per.tau.values[sel], delta=float(dlt))
        line.append(f"Δ={dlt}: {s['t_reco']:.1f}周")
    print(f"  {gn:10s}: " + " | ".join(line))
R["sens_delta"] = {gn: [group_schedule(per.tau.values[labs == k_], delta=float(dlt))["t_reco"]
                        for dlt in deltas] for gn, k_ in zip(final, [1, 2, 3])}

with open("solution/q2_results.json", "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1, default=str)
print("\n已保存 q2_results.json")

# ---------- 图 ----------
fig, ax = plt.subplots(figsize=(8.2, 5))
for gn, k_ in zip(final, [1, 2, 3]):
    sel = labs == k_
    s = group_schedule(per.tau.values[sel])
    ax.plot(GRID, s["rv"], lw=2, label=f"{gn} (n={sel.sum()})")
    ax.axvline(s["t_reco"], ls=":", color="gray", lw=1)
ax.axvline(12, color="r", ls="--", lw=.9, alpha=.6)
ax.set_xlabel("NIPT 时点 t (孕周)")
ax.set_ylabel("组内期望风险(相对权重)")
ax.set_title("期望风险-时点曲线 (阴影区=平带容差; 竖点线=推荐时点)")
ax.legend()
plt.tight_layout(); plt.savefig("figs/fig9_riskcurve.png"); plt.close()

fig, ax = plt.subplots(figsize=(8.2, 4.6))
colors = ["#4C72B0", "#55A868", "#C44E52"]
for k_, c in zip([1, 2, 3], colors):
    sel = labs == k_
    ax.hist(per.tau.values[sel], bins=np.arange(10, 31, .5), alpha=.55, color=c,
            label=list(final.keys())[k_-1])
ax.axvline(12, color="r", ls="--", lw=1)
ax.set_xlabel("个体最早达标时间 τ (孕周)"); ax.set_ylabel("人数")
ax.set_title("不同 BMI 组个体最早达标时间分布"); ax.legend()
plt.tight_layout(); plt.savefig("figs/fig10_tau_hist.png"); plt.close()

print("figs 已更新")
