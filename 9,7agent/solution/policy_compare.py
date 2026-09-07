# -*- coding: utf-8 -*-
"""政策对比表: 不同检测策略下的 期望风险/单次达标率/平均抽血次数/发现孕周分布"""
import sys, json, warnings
sys.path.insert(0, "solution")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from common import load_male

m = load_male().dropna(subset=["y_conc", "week", "bmi"])
import statsmodels.formula.api as smf
mm = smf.mixedlm("y_conc ~ week_c + bmi_c", data=m.assign(week_c=m.week - 11, bmi_c=m.bmi - 32.3),
                 groups=m["pid"], re_formula="~ week_c").fit(reml=True)
b0, bw, bb = mm.params["Intercept"], mm.params["week_c"], mm.params["bmi_c"]
rows = []
for pid, v in mm.random_effects.items():
    sub = m[m.pid == pid]
    tau = np.clip(11 + (0.04 - (b0 + v["Group"] + bb * (sub.bmi.iloc[0] - 32.3))) / (bw + v["week_c"]), 10, 30)
    rows.append(dict(pid=pid, bmi=sub.bmi.iloc[0], tau=float(tau)))
per = pd.DataFrame(rows)
K = np.log(4) / 9.0
D = 1.5
def risk(t, taus):
    d = np.where(taus <= t, t, taus + D)
    return float(np.mean(np.exp(K * (d - 11))))
def draws(t, taus):
    taus = np.asarray(taus)
    return float(np.mean(np.where(taus <= t, 1, 1 + (taus - t) / 2.0)))
def disc_fracs(t, taus):
    d = np.where(taus <= t, t, taus + D)
    return dict(le12=float(np.mean(d <= 12)), b13_27=float(np.mean((d > 12) & (d <= 27))),
                ge28=float(np.mean(d > 27)))

def ev(t, taus):
    return dict(t=float(t), risk=round(risk(t, taus), 3), pass_rate=round(float(np.mean(taus <= t)), 3),
                draws=round(draws(t, taus), 2), frac=disc_fracs(t, taus))

policies = {
    "A_统一11周": {"sel": per.tau.values, "t": 11.0},
    "A_统一10.5": {"sel": per.tau.values, "t": 10.5},
    "B_分组": {
        "g1_<36@11": {"sel": per.tau.values[per.bmi < 36], "t": 11.0},
        "g2_≥36@14.5": {"sel": per.tau.values[per.bmi >= 36], "t": 14.5}},
    "C_经验5档均12": {
        "g1_<28@12": {"sel": per.tau.values[per.bmi < 28], "t": 12.0},
        "g2_28-32@12": {"sel": per.tau.values[(per.bmi >= 28) & (per.bmi < 32)], "t": 12.0},
        "g3_32-36@12": {"sel": per.tau.values[(per.bmi >= 32) & (per.bmi < 36)], "t": 12.0},
        "g4_≥36@12": {"sel": per.tau.values[per.bmi >= 36], "t": 12.0}},
}
out = {}
print("=== 策略对比 ===")
for name, pl in policies.items():
    if "sel" in pl:
        r = ev(pl["t"], pl["sel"])
        print(f"{name}: t={r['t']} 风险={r['risk']} 达标率={r['pass_rate']} 平均抽血={r['draws']} {r['frac']}")
        out[name] = r
    else:
        tot = dict(n=0)
        wsum = {k: 0.0 for k in ["risk", "pass_rate", "draws", "le12", "b13_27", "ge28"]}
        parts = []
        for g, gg in pl.items():
            r = ev(gg["t"], gg["sel"]); n = len(gg["sel"])
            for k in wsum:
                wsum[k] += ({**{"risk": r["risk"], "pass_rate": r["pass_rate"], "draws": r["draws"]},
                              **r["frac"]}[k]) * n
            tot["n"] += n
            parts.append({g: r})
        avg = {k: round(v / tot["n"], 3) for k, v in wsum.items()}
        print(f"{name}: 加权平均 风险={avg['risk']} 达标率={avg['pass_rate']} 抽血={avg['draws']} "
              f"发现≤12={avg['le12']} 13-27={avg['b13_27']} ≥28={avg['ge28']}")
        out[name] = {"parts": parts, "avg": avg}

# 决策对边界BMI的个体示例: 推荐时点随BMI
print("\n=== 不同BMI个体的单次达标时间与推荐解读 ===")
for b in [28, 30, 32, 34, 36, 38, 40]:
    taus = per.tau.values[per.bmi < b + 0.5]  # 仅展示尺度
    pass_at_11 = float(np.mean(taus <= 11))
    print(f"BMI<{b}: ...")
for b in [28, 30, 32, 34]:
    taus = per.tau.values[(per.bmi >= b - 1) & (per.bmi < b + 1)]
    taus2 = per.tau.values[(per.bmi >= 36) & (per.bmi < 38)]
    if len(taus) == 0: continue
    print(f"BMI∈[{b-1},{b+1}): n={len(taus)} P50={np.median(taus):.2f} P90={np.quantile(taus,.9):.2f} | "
          f"BMI∈[36,38): n={len(taus2)} P50={np.median(taus2):.2f}")
json.dump(out, open("solution/政策对比.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("\n已存 solution/政策对比.json")
