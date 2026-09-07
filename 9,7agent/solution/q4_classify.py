# -*- coding: utf-8 -*-
"""
问题4: 女胎染色体非整倍体(13/18/21号)判定方法
- 标签: AB列(非整倍体) 行级/孕妇级
- 特征: 各染色体Z值、X染色体Z值与浓度、GC含量(总及各染色体)、读段数及
        比对/重复/过滤比例、BMI、年龄、孕周等
- 流程:
  1) 数据核查(标签与z规则一致性、质控指标分布)
  2) 质控(QC)门控: GC范围/比对率/重复率/唯一比对读段数 -> 判定样本有效性
  3) 行级判定模型: LR(带惩罚, 组CV), RF 对照; 每染色体分别建模 + 综合判定
  4) 孕妇级整合: 多次检测证据合并(取最大异常概率/次数表决), 评估提升
  5) 阈值选择(Youden), 混淆矩阵, 校准, 特征解释, 灵敏度分析
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
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             confusion_matrix, precision_recall_curve, brier_score_loss)
from common import load_female

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

f = load_female().dropna(subset=["bmi", "age", "week"])
f["ab"] = f.aneu.fillna("")
for ch, ab in [("13", "ab13"), ("18", "ab18"), ("21", "ab21")]:
    f[ab] = f.ab.str.contains("T" + ch).astype(int)
f["any_ab"] = (f.ab != "").astype(int)
f["reads_ratio"] = f.uniq_reads / f.raw_reads
R = {}
print(f"女胎样本 {len(f)} 行 / {f.pid.nunique()} 孕妇; 异常行 {f.any_ab.sum()} ({f.any_ab.mean()*100:.1f}%)")

# ============ 1) 数据核查 ============
print("\n== 1. 标签-测序特征一致性核查 ==")
for ch, zc, ab in [("13", "z13", "ab13"), ("18", "z18", "ab18"), ("21", "z21", "ab21")]:
    pos = f[ab] == 1
    print(f"chr{ch}: 阳性行n={pos.sum()}, 阳性z中位={f.loc[pos, zc].median():.2f}, z>3占比={(f.loc[pos,zc]>3).mean()*100:.0f}%"
          f" | 阴性行z>3的行数={int((f.loc[~pos, zc]>3).sum())}")
R["label_check"] = {ch: dict(pos_n=int((f[ab]==1).sum())) for ch, ab in zip("131821".replace("1","")+"", [])}  # placeholder
# 孕妇级标签一致性
piv = f.groupby("pid").ab.agg(lambda s: len(set(s)))
print(f"多次检测标签不一致孕妇: {(piv>1).sum()}/{len(piv)}")
R["label_consistency"] = dict(rows=len(f), women=f.pid.nunique(), pos_rows=int(f.any_ab.sum()),
                              inconsistent_women=int((piv > 1).sum()))

# ============ 2) QC 门控 ============
# 采用临床常用 NIPT 质控: GC∈[0.40,0.60], 比对率≥0.70(数据内约0.8), 唯一比对≥200万, 重复率≤0.30
f["qc_ok"] = ((f.gc >= 0.40) & (f.gc <= 0.60) &
              (f.map_ratio >= 0.70) & (f.dup_ratio <= 0.30) &
              (f.uniq_reads >= 2e6)).astype(int)
print(f"\n== 2. QC 门控: 通过 {f.qc_ok.sum()}/{len(f)} ({f.qc_ok.mean()*100:.1f}%), "
      f"未通过中异常行占比 {(f.loc[f.qc_ok==0,'any_ab'].mean()*100 if (f.qc_ok==0).any() else 0):.1f}% (通过者 {(f.loc[f.qc_ok==1,'any_ab'].mean()*100):.1f}%)")
R["qc"] = dict(pass_n=int(f.qc_ok.sum()), total=int(len(f)))

# ============ 3) 行级判定模型 ============
feats = ["z13", "z18", "z21", "zx", "x_conc", "gc", "gc13", "gc18", "gc21", "bmi",
         "age", "week", "raw_reads", "uniq_reads", "map_ratio", "dup_ratio", "flt_ratio"]
X = f[feats].copy()
X["x_conc_sq"] = X.x_conc ** 2          # 负值深度信息
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
gkp = GroupKFold(n_splits=5)

def cv_run(yvec, model_fn, scaler=True):
    oof = np.zeros(len(X))
    for tr, te in gkp.split(X, yvec, f.pid.values):
        if scaler:
            sc = StandardScaler().fit(X.iloc[tr])
            Xtr, Xte = sc.transform(X.iloc[tr]), sc.transform(X.iloc[te])
        else:
            Xtr, Xte = X.iloc[tr], X.iloc[te]
        mdl = model_fn().fit(Xtr, yvec.iloc[tr])
        oof[te] = mdl.predict_proba(Xte)[:, 1]
    return oof

print("\n== 3. 行级判定模型 (GroupKFold 5折, 按孕妇分组) ==")
targets = {"任意非整倍体": f.any_ab, "T13": f.ab13, "T18": f.ab18, "T21": f.ab21}
res_rows = {}
for tname, tv in targets.items():
    if tv.sum() < 5:
        continue
    oof_lr = cv_run(tv, lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
    oof_rf = cv_run(tv, lambda: RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                                       min_samples_leaf=3, n_jobs=-1, random_state=3))
    auc_lr = roc_auc_score(tv, oof_lr)
    auc_rf = roc_auc_score(tv, oof_rf)
    apr_lr = average_precision_score(tv, oof_lr)
    res_rows[tname] = dict(auc_lr=float(auc_lr), auc_rf=float(auc_rf), apr_lr=float(apr_lr),
                           n_pos=int(tv.sum()))
    print(f"  {tname:8s} (阳性行n={tv.sum():3d}): LR OOF AUC={auc_lr:.3f} | RF OOF AUC={auc_rf:.3f} | LR AP={apr_lr:.3f}")
R["row_models"] = res_rows

# 任意异常: 保存OOF概率, 阈值分析
oof_any = cv_run(f.any_ab, lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
f["p_abn"] = oof_any
# Youden 阈值
fpr_, tpr_, thr_ = roc_curve(f.any_ab, oof_any)
youden = thr_[np.argmax(tpr_ - fpr_)]
prec_, rec_, thrp = precision_recall_curve(f.any_ab, oof_any)
# 平衡阈值≈0.5下的混淆
thr_use = 0.5
cm = confusion_matrix(f.any_ab, (oof_any >= thr_use).astype(int))
tn, fp_, fn_, tp_ = cm.ravel()
print(f"\n  [任意异常] Youden阈值={youden:.3f}; 阈值0.5: 敏感性={tp_/(tp_+fn_):.3f} 特异性={tn/(tn+fp_):.3f} "
      f"精确率={tp_/(tp_+fp_):.3f} (阳性行n={f.any_ab.sum()})")
R["youden"] = float(youden)
R["cm_0.5"] = dict(tn=int(tn), fp=int(fp_), fn=int(fn_), tp=int(tp_))

# 校准
bs = brier_score_loss(f.any_ab, oof_any)
base_brier = f.any_ab.mean() * (1 - f.any_ab.mean())
print(f"  Brier={bs:.4f} (基准={base_brier:.4f})")
R["brier"] = float(bs)

# LR 系数(全数据拟合, 标准化后, 供解释)
sc = StandardScaler().fit(X)
lr_all = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(sc.transform(X), f.any_ab)
coef = pd.Series(lr_all.coef_[0], index=X.columns).sort_values()
print("\n  LR 系数(标准化特征, 降序):")
print((coef.sort_values(ascending=False).round(3)).to_string())
R["lr_coef"] = coef.round(3).to_dict()

# ============ 4) 孕妇级整合 ============
print("\n== 4. 孕妇级证据整合 (多次检测合并) ==")
# 策略: 该孕妇任意一次判定阳性即转诊/复核 (临床筛查语义)
per = f.groupby("pid").agg(pmax=("p_abn", "max"), pmean=("p_abn", "mean"),
                           pn=("p_abn", "size"), any_ab=("any_ab", "max"),
                           any_ab_sum=("any_ab", "sum"), ab=("ab", "first"))
per["pos_anyrow"] = (per.pmax >= thr_use).astype(int)
per["pos_mean"] = (per.pmean >= thr_use).astype(int)
# 只保留多行证据整合
auc_max = roc_auc_score(per.any_ab, per.pmax)
auc_mean = roc_auc_score(per.any_ab, per.pmean)
print(f"  孕妇级 (n={len(per)}, 阳性孕妇={int(per.any_ab.sum())}): AUC(max)={auc_max:.3f} AUC(mean)={auc_mean:.3f}")
cm_w = confusion_matrix(per.any_ab, per.pos_anyrow).ravel()
tn, fp_, fn_, tp_ = cm_w
print(f"  [孕妇级 · 任一阳性即判阳性] 敏感性={tp_/(tp_+fn_):.3f} 特异性={tn/(tn+fp_):.3f}")
print(f"  假阳性孕妇中 平均阳性行数={per.loc[(per.any_ab==0)&(per.pos_anyrow==1), 'any_ab_sum'].mean():.2f} (均为假阳性行)")
R["woman_level"] = dict(auc_max=float(auc_max), auc_mean=float(auc_mean), n=int(len(per)))

# 检出灵敏度 vs 复检次数: 只用前k行检测时的性能 (第1/2/3/4次检测)
print("\n  -- 随检测次数累积的判定性能 (只用前k次检测) --")
for k in [1, 2, 3, 4]:
    sub = f.sort_values(["pid", "week"]).groupby("pid").head(k)
    if sub.any_ab.sum() == 0 or (sub.any_ab == 0).sum() == 0:
        continue
    a = roc_auc_score(sub.any_ab, sub.p_abn)
    print(f"    前{k}次检测: 行数{len(sub)}, AUC={a:.3f}")
R["draws_k"] = {}

# ============ 5) 灵敏度/误差分析 ============
print("\n== 5. 误差与稳健性 ==")
# (a) 概率阈值滑动: 敏感性/特异性
ths = np.arange(0.2, 0.9, 0.1)
rows_th = []
for th in ths:
    cmt = confusion_matrix(f.any_ab, (oof_any >= th).astype(int)).ravel()
    tn, fp_, fn_, tp_ = cmt
    rows_th.append(dict(th=round(float(th), 2), sens=round(tp_/(tp_+fn_), 3),
                        spec=round(tn/(tn+fp_), 3)))
print(pd.DataFrame(rows_th).round(3).to_string(index=False))
R["threshold_table"] = rows_th
# (b) 自助法 AUC 区间 (按孕妇重采样)
rng = np.random.default_rng(5)
aucs_b = []
for b in range(400):
    idx = rng.integers(0, len(per), len(per))
    if per.any_ab.iloc[idx].nunique() < 2:
        continue
    a = roc_auc_score(per.any_ab.iloc[idx], per.pmax.iloc[idx])
    aucs_b.append(a)
aucs_b = np.array(aucs_b)
print(f"  孕妇级 AUC 自助95%CI: [{np.quantile(aucs_b, .025):.3f}, {np.quantile(aucs_b, .975):.3f}]")
R["auc_ci"] = [float(np.quantile(aucs_b, .025)), float(np.quantile(aucs_b, .975))]
# (c) 特征扰动: 删除某一组特征后 AUC 变化(消融)
print("\n  -- 特征消融 (LR OOF AUC, 任意异常) --")
base_auc = roc_auc_score(f.any_ab, oof_any)
for grp, cols in [("Z值组", ["z13", "z18", "z21"]), ("X染色体组", ["zx", "x_conc", "x_conc_sq"]),
                  ("GC组", ["gc", "gc13", "gc18", "gc21"]), ("读段/质控组", ["raw_reads", "uniq_reads", "map_ratio", "dup_ratio", "flt_ratio"]),
                  ("孕妇特征", ["bmi", "age", "week"])]:
    rest = [c for c in X.columns if c not in cols]
    oof_r = np.zeros(len(X))
    for tr, te in gkp.split(X, f.any_ab, f.pid.values):
        Xr = X.iloc[tr][rest]
        sc2 = StandardScaler().fit(Xr)
        mm = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(sc2.transform(Xr), f.any_ab.iloc[tr])
        oof_r[te] = mm.predict_proba(sc2.transform(X.iloc[te][rest]))[:, 1]
    a_ab = roc_auc_score(f.any_ab, oof_r)
    print(f"    去掉{grp:14s}: AUC={a_ab:.3f} (下降 {base_auc-a_ab:+.3f})")
R["ablation"] = dict(base=float(base_auc))

# ============ 图 ============
fig, ax = plt.subplots(figsize=(7.4, 5.6))
fpr_all, tpr_all, _ = roc_curve(f.any_ab, oof_any)
ax.plot(fpr_all, tpr_all, lw=2.2, label=f"行级 LR (AUC={roc_auc_score(f.any_ab, oof_any):.3f})")
fpr_w, tpr_w, _ = roc_curve(per.any_ab, per.pmax)
ax.plot(fpr_w, tpr_w, lw=2.2, label=f"孕妇级 max 整合 (AUC={auc_max:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5)
ax.set_xlabel("假阳性率 (1-特异性)"); ax.set_ylabel("真阳性率 (敏感性)")
ax.set_title("女胎非整倍体判定模型 ROC")
ax.legend()
plt.tight_layout(); plt.savefig("figs/fig13_roc.png"); plt.close()

fig, ax = plt.subplots(figsize=(7.4, 5.2))
coef_sorted = coef.sort_values()
colors = ["#C44E52" if v < 0 else "#4C72B0" for v in coef_sorted]
ax.barh(coef_sorted.index, coef_sorted.values, color=colors)
ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("标准化系数")
ax.set_title("LR 判定模型特征系数 (正=增加异常概率)")
plt.tight_layout(); plt.savefig("figs/fig14_coef.png"); plt.close()

# 校准曲线
fig, ax = plt.subplots(figsize=(6.6, 5))
from sklearn.calibration import calibration_curve
pc_true, pc_pred = calibration_curve(f.any_ab, oof_any, n_bins=8)
ax.plot(pc_pred, pc_true, "-o", label="LR 判定模型")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("预测异常概率"); ax.set_ylabel("实际异常比例")
ax.set_title("校准曲线")
ax.legend()
plt.tight_layout(); plt.savefig("figs/fig15_calib.png"); plt.close()

with open("solution/q4_results.json", "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1, default=str)
f[["pid", "draw", "week", "any_ab"] + ["p_abn"]].to_csv("solution/女胎行级判定概率.csv", index=False)
per[["pmax", "pmean", "any_ab"]].to_csv("solution/女胎孕妇级判定概率.csv")
print("\n已保存 q4_results.json + figs/fig13-15 + 概率表")
