# -*- coding: utf-8 -*-
"""Q4b: 校准 + 分染色体判定表 + 孕妇级 k-次阳性复检规则 + 交叉验证细节"""
import sys, os, json, warnings
sys.path.insert(0, "solution")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             average_precision_score, brier_score_loss)
from common import load_female, load_male

raw = load_female().dropna(subset=["bmi", "age", "week"]).reset_index(drop=True)
f = raw.copy()
f["ab"] = f.aneu.fillna("")
for ch, ab in [("13", "ab13"), ("18", "ab18"), ("21", "ab21")]:
    f[ab] = f.ab.str.contains("T" + ch).astype(int)
f["any_ab"] = (f.ab != "").astype(int)

feats = ["z13", "z18", "z21", "zx", "x_conc", "gc", "gc13", "gc18", "gc21", "bmi",
         "age", "week", "raw_reads", "uniq_reads", "map_ratio", "dup_ratio", "flt_ratio"]
X = f[feats].copy()
X["x_conc_sq"] = X.x_conc ** 2
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
gkp = GroupKFold(n_splits=5)

def cv_oof(y):
    oof = np.zeros(len(X))
    for tr, te in gkp.split(X, y, f.pid.values):
        sc = StandardScaler().fit(X.iloc[tr])
        m = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)
        m.fit(sc.transform(X.iloc[tr]), y.iloc[tr])
        oof[te] = m.predict_proba(sc.transform(X.iloc[te]))[:, 1]
    return oof

out = {}
# 分染色体工作点
print("== 分染色体判定工作点 (OOF, Youden) ==")
det = {}
for ch, ab in [("13", "ab13"), ("18", "ab18"), ("21", "ab21")]:
    y = f[ab]
    p = cv_oof(y)
    fpr, tpr, thr = roc_curve(y, p)
    j = int(np.argmax(tpr - fpr))
    th = float(thr[j])
    cm = confusion_matrix(y, (p >= th).astype(int)).ravel()
    tn, fp_, fn_, tp_ = cm
    det[ch] = dict(th=round(th, 3), auc=round(float(roc_auc_score(y, p)), 3),
                   sens=round(tp_ / (tp_ + fn_), 3), spec=round(tn / (tn + fp_), 3),
                   pos_n=int(y.sum()))
    print(f"  chr{ch}: Youden阈值={th:.3f} AUC={det[ch]['auc']} 敏感性={det[ch]['sens']} 特异性={det[ch]['spec']}")
out["per_chr"] = det

# 综合判定: 三染色体 OR (各自阈值) vs 行级
y_any = f.any_ab
p13 = cv_oof(f.ab13); p18 = cv_oof(f.ab18); p21 = cv_oof(f.ab21)
pred_or = ((p13 >= det["13"]["th"]) | (p18 >= det["18"]["th"]) | (p21 >= det["21"]["th"])).astype(int)
cm = confusion_matrix(y_any, pred_or).ravel()
tn, fp_, fn_, tp_ = cm
print(f"\n[三染色体OR规则] 敏感性={tp_/(tp_+fn_):.3f} 特异性={tn/(tn+fp_):.3f} 精确率={tp_/(tp_+fp_):.3f}")
out["or_rule"] = dict(sens=round(tp_/(tp_+fn_), 3), spec=round(tn/(tn+fp_), 3), prec=round(tp_/(tp_+fp_), 3))

# 校准: isotonic (OOF上拟合再应用)
p_any = cv_oof(y_any)
iso = IsotonicRegression(out_of_bounds="clip").fit(p_any, y_any)
p_cal = iso.predict(p_any)
bs0 = brier_score_loss(y_any, p_any); bs1 = brier_score_loss(y_any, p_cal)
print(f"Brier: 原始={bs0:.4f} → 校准后={bs1:.4f} (基准={y_any.mean()*(1-y_any.mean()):.4f})")
out["brier"] = dict(raw=float(bs0), cal=float(bs1))

# 孕妇级 k-of-n 复检规则
per = pd.DataFrame({"pid": f.pid.values, "p": p_cal, "ab": y_any.values}).groupby("pid").agg(
    n=("p", "size"), pmax=("p", "max"), pos=("ab", "max"), n_pos=("ab", "sum"))
print("\n== 孕妇级复检规则 (校准后概率) ==")
for k in [1, 2, 3]:
    for th in [0.5, 0.6, 0.7]:
        flag = ((per.n >= k) & (per.pmax >= th)).astype(int) if k == 1 else \
               ((per.n >= k) & (per.n_pos >= k)).astype(int) if False else None
# 规则A: 任意一次 p>0.5; B: ≥2次 p>0.5; C: 任意一次 p>0.7
for name, rule in [("A: 任一次p>0.5", per.pmax > 0.5),
                   ("B: ≥2次p>0.5", per.n_pos >= 2),
                   ("C: 任一次p>0.7", per.pmax > 0.7)]:
    cm = confusion_matrix(per.pos, rule.astype(int)).ravel()
    tn, fp_, fn_, tp_ = cm
    print(f"  {name:16s}: 敏感性={tp_/(tp_+fn_):.3f} 特异性={tn/(tn+fp_):.3f} "
          f"PPV={tp_/(tp_+fp_):.3f} (阳性孕妇{int(tp_)}+假阳{int(fp_)})")
# 阈值-敏感/特异 (校准后)
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    cm = confusion_matrix(per.pos, (per.pmax >= th).astype(int)).ravel()
    tn, fp_, fn_, tp_ = cm
    print(f"  孕妇级阈值{th:.1f}: 敏感性={tp_/(tp_+fn_):.3f} 特异性={tn/(tn+fp_):.3f}")

# 数据核查补充: T21 判阳性标签行在男胎/女胎中的特征
print("\n== T21 标签行的染色体21相关特征 (男/女胎) ==")
for sex, df in [("女胎", f), ("男胎", load_male())]:
    df = df.copy(); df["ab"] = df.aneu.fillna(""); df["t21"] = df.ab.str.contains("T21").astype(int)
    if df.t21.sum() == 0: continue
    d = df[df.t21 == 1]
    e = df[df.t21 == 0]
    print(f"  {sex}: T21行n={len(d)}: z21中位 {d.z21.median():.2f} vs 阴性 {e.z21.median():.2f}; "
          f"gc21中位 {d.gc21.median():.4f} vs {e.gc21.median():.4f}")
    if "x_conc" in d:
        print(f"     x_conc中位 {d.x_conc.median():.4f} vs {e.x_conc.median():.4f}")

with open("solution/q4b_results.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print("\nsaved q4b_results.json")
