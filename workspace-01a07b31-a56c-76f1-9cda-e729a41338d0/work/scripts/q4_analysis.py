# -*- coding: utf-8 -*-
"""问题4：女胎异常判定方法。

以女胎数据 AB 列（13/18/21 号染色体非整倍体检出）为判定结果标签：
  1) 基线规则：|Z_k| >= 3（经典阈值）及各阈值变体；
  2) 单因素与多因素 logistic 回归（每条染色体），稳健标准误；
  3) 机器学习（随机森林/梯度提升）二分类"任一异常"与逐染色体分类，
     按孕妇分组的分层交叉验证（避免同一孕妇多次检测泄漏）；
  4) 特征重要性、ROC、最终判定规则与阈值。
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import statsmodels.api as sm

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from common import load_female, clean_female, WORK

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib_templates import forest, calibration_residuals, save_figure

import figure_style  # noqa
FIG = os.path.join(WORK, 'figures')
RES = os.path.join(WORK, 'results')

FEATS = ['z13', 'z18', 'z21', 'zx', 'conc_x', 'gc', 'gc13', 'gc18', 'gc21',
         'bmi', 'gest', 'age', 'reads_total', 'map_ratio', 'dup_ratio',
         'filter_ratio']


def parse_labels(df):
    a = df['aneuploidy'].fillna('')
    df = df.copy()
    df['y_any'] = (a != '').astype(int)
    df['y13'] = a.str.contains('T13').astype(int)
    df['y18'] = a.str.contains('T18').astype(int)
    df['y21'] = a.str.contains('T21').astype(int)
    return df


def cv_eval(model_fn, X, y, groups, n_splits=5, seed=0):
    """按孕妇分组的分层 CV，返回合并后的预测概率/标签与指标。"""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = np.full(len(y), np.nan)
    pred = np.full(len(y), -1)
    for tr, te in sgkf.split(X, y, groups):
        m = model_fn()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        proba[te] = p
        pred[te] = (p >= 0.5).astype(int)
    auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan
    return proba, pred, auc


def main():
    df, log = clean_female(load_female())
    df = parse_labels(df)
    out = {'n': len(df), 'n_mothers': int(df['mother'].nunique()),
           'n_abnormal_rows': int(df['y_any'].sum()),
           'label_dist': {'any': int(df['y_any'].sum()),
                           'T13': int(df['y13'].sum()), 'T18': int(df['y18'].sum()),
                           'T21': int(df['y21'].sum())}}

    # ---------- 单因素差异 ----------
    uni = {}
    for c in FEATS:
        a = df[df['y_any'] == 1][c].values
        b = df[df['y_any'] == 0][c].values
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        uni[c] = {'mean_abn': round(float(np.mean(a)), 4),
                  'mean_norm': round(float(np.mean(b)), 4), 'p': f'{p:.2e}'}
    out['univariate'] = uni

    # ---------- 基线规则：|Z|>=阈值 ----------
    base = {}
    for th in [2.0, 2.5, 3.0]:
        pred13 = (df['z13'].abs() >= th).astype(int)
        pred18 = (df['z18'].abs() >= th).astype(int)
        pred21 = (df['z21'].abs() >= th).astype(int)
        pred_any = ((pred13 + pred18 + pred21) > 0).astype(int)
        # 逐染色体匹配
        def perf(p, y):
            cm = confusion_matrix(y, p, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            return {'sens': round(tp / max(tp + fn, 1), 3),
                    'spec': round(tn / max(tn + fp, 1), 3),
                    'acc': round(accuracy_score(y, p), 3)}
        base[f'Z>={th}'] = {'any': perf(pred_any, df['y_any'].values),
                            'T13': perf(pred13, df['y13'].values),
                            'T18': perf(pred18, df['y18'].values),
                            'T21': perf(pred21, df['y21'].values)}
    out['baseline_Z_rules'] = base

    # ---------- 逐染色体 logistic（稳健） ----------
    Xdf = df[FEATS].copy()
    Xdf['zx_abs'] = df['zx'].abs()
    logit_res = {}
    for k, zcol in [('T13', 'z13'), ('T18', 'z18'), ('T21', 'z21')]:
        y = df[['y13', 'y18', 'y21'][['T13', 'T18', 'T21'].index(k)]].values
        base_f = [zcol, 'zx', 'conc_x', 'gc', 'bmi', 'gest']
        X = sm.add_constant(df[base_f].astype(float).values)
        try:
            m = sm.Logit(y, X).fit(disp=0, cov_type='HC1')
            logit_res[k] = {
                'features': ['const'] + base_f,
                'params': {f: {'coef': round(float(c), 4),
                               'OR': round(float(np.exp(c)), 3),
                               'p': f"{float(p):.2e}"}
                           for f, c, p in zip(['const'] + base_f, m.params, m.pvalues)}}
        except Exception as e:
            logit_res[k] = {'error': str(e)}
    out['logistic_per_chr'] = logit_res

    # ---------- 机器学习：任一异常 + 逐染色体 ----------
    X = Xdf.values
    groups = df['mother'].values
    results = {}
    roc_data = {}

    def rf():
        return Pipeline([('sc', StandardScaler()),
                         ('rf', RandomForestClassifier(500, max_depth=6,
                                                       class_weight='balanced',
                                                       random_state=0, n_jobs=2))])

    def gb():
        return Pipeline([('sc', StandardScaler()),
                         ('gb', GradientBoostingClassifier(random_state=0))])

    def lr():
        return Pipeline([('sc', StandardScaler()),
                         ('lr', LogisticRegression(max_iter=2000, C=1.0))])

    targets = [('any', df['y_any'].values), ('T13', df['y13'].values),
               ('T18', df['y18'].values), ('T21', df['y21'].values)]
    proba_store = {}
    for tname, y in targets:
        res = {}
        probas = {}
        for mname, fn in [('LR', lr), ('RF', rf), ('GB', gb)]:
            if len(np.unique(y)) < 2:
                continue
            proba, pred, auc = cv_eval(fn, X, y, groups)
            probas[mname] = proba
            proba_store.setdefault(tname, {})[mname] = proba
            # Youden 阈值
            fpr, tpr, ths = roc_curve(y, proba)
            j = np.argmax(tpr - fpr)
            th_y = float(ths[j])
            pred_y = (proba >= th_y).astype(int)
            cm = confusion_matrix(y, pred_y, labels=[0, 1])
            tn, fp, fnn, tp = cm.ravel()
            res[mname] = {'auc': round(float(auc), 3),
                          'youden_thr': round(th_y, 3),
                          'sens': round(tp / max(tp + fnn, 1), 3),
                          'spec': round(tn / max(tn + fp, 1), 3),
                          'acc': round(accuracy_score(y, pred_y), 3)}
        results[tname] = res
        best_m = max(res, key=lambda kk: res[kk]['auc'])
        fpr, tpr, _ = roc_curve(y, probas[best_m])
        roc_data[tname] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
                           'auc': res[best_m]['auc'], 'model': best_m}
    out['ml_cv'] = results
    # 特征重要性（RF，任一异常，全量拟合）
    rf_full = rf().fit(X, df['y_any'].values)
    imp = rf_full.named_steps['rf'].feature_importances_
    out['rf_importance_any'] = sorted(
        zip(Xdf.columns, [round(float(v), 4) for v in imp]), key=lambda z: -z[1])

    # ---------- 最终判定规则（标准化特征的 logistic 分数，全部数据拟合） ----------
    y = df['y_any'].values
    mu = Xdf.mean(axis=0).values
    sd = Xdf.std(axis=0).values
    Xstd = (Xdf.values - mu) / sd
    Xf = sm.add_constant(Xstd)
    mfinal = sm.Logit(y, Xf).fit(disp=0, cov_type='HC1')
    out['final_rule'] = {
        'features': ['const'] + Xdf.columns.tolist(),
        'std_mean': {f: round(float(a), 6) for f, a in zip(Xdf.columns, mu)},
        'std_sd': {f: round(float(a), 6) for f, a in zip(Xdf.columns, sd)},
        'params': {f: {'coef': round(float(c), 4), 'p': f"{float(p):.2e}"}
                   for f, c, p in zip(['const'] + Xdf.columns.tolist(),
                                      mfinal.params, mfinal.pvalues)},
        'note': '得分 s = β0 + Σ β_k (x_k-μ_k)/σ_k；概率 = 1/(1+e^-s)，阈值见 Youden'}
    proba_full = mfinal.predict(Xf)
    fpr, tpr, ths = roc_curve(y, proba_full)
    j = np.argmax(tpr - fpr)
    out['final_rule']['full_auc'] = round(float(roc_auc_score(y, proba_full)), 3)
    out['final_rule']['youden_thr'] = round(float(ths[j]), 3)
    out['final_rule']['full_sens'] = round(float(tpr[j]), 3)
    out['final_rule']['full_spec'] = round(float(1 - fpr[j]), 3)

    # ---------- 图 ----------
    # 图A：各目标最佳模型 ROC
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    colors = {'any': 'crimson', 'T13': '#4878CF', 'T18': '#6AACB8', 'T21': '#EE854A'}
    for tname, rd in roc_data.items():
        ax.plot(rd['fpr'], rd['tpr'], color=colors[tname], lw=2,
                label=f"{tname}（{rd['model']}, AUC={rd['auc']:.2f}）")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('1 - 特异度'); ax.set_ylabel('灵敏度')
    ax.set_title('女胎非整倍体判定 ROC（按孕妇分组 CV）')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q4_roc'))

    # 图B：logistic OR 森林图（最终规则，任一异常）
    names = [f for f in out['final_rule']['features'] if f != 'const']
    coefs = np.array([out['final_rule']['params'][f]['coef'] for f in names])
    # 近似 95% CI（用 HC1 标准误）
    se = mfinal.bse[1:]
    lo, hi = coefs - 1.96 * se, coefs + 1.96 * se
    fig = forest(np.exp(coefs), np.exp(lo), np.exp(hi), names,
                 interval_label='95% CI', ratio=True)
    fig.suptitle('女胎异常判定 logistic 回归比值比（任一非整倍体）', fontsize=12)
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q4_or'))

    # 图C：校准图（任一异常，最佳模型的 CV 概率）
    best_any = max(results['any'], key=lambda kk: results['any'][kk]['auc'])
    fig = calibration_residuals(df['y_any'].values, proba_store['any'][best_any],
                                task='classification')
    fig.suptitle(f'女胎异常判定校准图（{best_any}，分组 CV 概率）', fontsize=12)
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q4_calibration'))

    # 图D：Z 值散点（13/18/21）按标签着色
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (k, zc) in zip(axes, [('T13', 'z13'), ('T18', 'z18'), ('T21', 'z21')]):
        nrm = df[df[f'y{k[1:]}'] == 0]
        abn = df[df[f'y{k[1:]}'] == 1]
        ax.scatter(nrm.index, nrm[zc], s=8, alpha=0.4, color='#6AACB8', label='未检出')
        ax.scatter(abn.index, abn[zc], s=20, color='crimson', label='检出', marker='^')
        ax.axhline(3, color='gray', ls='--', lw=1)
        ax.axhline(-3, color='gray', ls='--', lw=1)
        ax.set_title(f'{k}: {zc}')
        ax.set_xlabel('样本行号')
    axes[0].set_ylabel('Z 值'); axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q4_z'))

    with open(os.path.join(RES, 'q4_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    np.savez(os.path.join(RES, 'q4_roc.npz'),
             **{f'{k}_{kk}': np.array(v[kk]) for k, v in roc_data.items()
                for kk in ['fpr', 'tpr']})
    print('OK q4')


if __name__ == '__main__':
    main()
