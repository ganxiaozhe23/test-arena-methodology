# -*- coding: utf-8 -*-
"""问题3：综合身高、体重、年龄等因素的男胎孕妇分组与最佳 NIPT 时点。

在问题1/2框架上：
  1) 扩展回归：lnC ~ gest + weight + height + age（及备选形式），选择并检验显著性；
     混合效应版本给出个体内纵向效应。
  2) 个体化达标概率 p_i(t)=Φ((μ_i(t)-ln0.04)/s)，μ 由扩展模型给出。
  3) 仍按 BMI 分组（DP，最小样本量约束），组内共用时点最小化 Σ 个体期望风险；
     同时给出个体化时点 t_i* 及其相对组内统一时点的风险改进。
  4) 达标比例：模型预测与经验值对照；检测误差 σ_e 的敏感性。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(__file__))
from common import load_male, clean_male, WORK
from q2_analysis import (T_GRID, DELTA, LAMBDA, CFG, risk_w, p_seqfail,
                         dp_partition, BMI_RES, N_MIN, THRESH)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib_templates import forest, save_figure

import figure_style  # noqa
FIG = os.path.join(WORK, 'figures')
RES = os.path.join(WORK, 'results')


def main():
    df, log = clean_male(load_male())
    df['lnC'] = np.log(df['conc_y'])
    out = {'n': len(df)}

    # ---------- 扩展模型选择 ----------
    jac = np.log(df['conc_y'].values).sum()
    cand = {
        'E1_q1+age': 'lnC ~ gest + bmi + I(bmi**2) + age',
        'E2_wh+age': 'lnC ~ gest + weight + height + age',
        'E3_wh+age+bmi': 'lnC ~ gest + weight + height + age + bmi',
        'E4_wh_quad': 'lnC ~ gest + weight + height + age + I(weight**2)',
    }
    llC, mods = {}, {}
    for name, f in cand.items():
        m = smf.ols(f, data=df).fit(cov_type='HC3')
        mods[name] = m
        llC[name] = m.llf - jac
        out.setdefault('models', {})[name] = {
            'formula': f, 'R2': round(m.rsquared, 4), 'AIC_c': round(m.aic, 1),
            'BIC': round(m.bic, 1),
            'params': {k: {'coef': round(float(v), 5), 'p': f"{float(m.pvalues[k]):.2e}"}
                       for k, v in m.params.items()}}
    best = max(llC, key=lambda k: llC[k] - len(mods[k].params) * np.log(len(df)) / 2)
    out['selected'] = best
    out['llC'] = {k: round(v, 1) for k, v in llC.items()}
    m = mods[best]

    # VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = m.model.exog
    out['VIF'] = {name: round(float(variance_inflation_factor(X, i)), 2)
                  for i, name in enumerate(m.model.exog_names)}

    # 混合效应（个体内效应）
    mm = smf.mixedlm(m.model.formula.replace('lnC ~', 'lnC ~'), df,
                     groups=df['mother']).fit(reml=False)
    out['mixed'] = {k: {'coef': round(float(v), 5), 'p': f"{float(mm.pvalues[k]):.2e}"}
                    for k, v in mm.params.items() if k != 'Group'}
    out['mixed_var'] = {'intercept_var': round(float(mm.cov_re.iloc[0, 0]), 5),
                        'resid_var': round(float(mm.scale), 5)}
    s2_ext = out['mixed_var']['intercept_var'] + out['mixed_var']['resid_var']

    # ---------- 个体化风险曲线 ----------
    exog = pd.DataFrame(m.model.exog, columns=m.model.exog_names)
    beta = m.params.values

    def mu_ind(t, sub_exog):
        """t: 标量; sub_exog: n×p（除截距列外替换为 t 的取值）"""
        X = sub_exog.copy()
        if 'gest' in X.columns:
            X['gest'] = t
        return X.values @ beta

    n = len(df)
    RM_ind = np.empty((n, len(T_GRID)))
    NT_ind = np.empty((n, len(T_GRID)))
    pmat = np.empty((n, len(T_GRID)))
    for j, t in enumerate(T_GRID):
        mu = mu_ind(t, exog)
        p = stats.norm.cdf((mu - np.log(THRESH)) / np.sqrt(s2_ext))
        pmat[:, j] = p
        pv = p * (1 - p_seqfail(t))
        w = risk_w(t)
        RM_ind[:, j] = pv * w  # 递推部分后面补
    # 反向递推（个体）
    step = int(round(DELTA / 0.25))
    ext_t = np.arange(T_GRID.min(), 30.0 + DELTA + 1e-9, 0.25)
    PV_ext = np.empty((n, len(ext_t)))
    for j, t in enumerate(ext_t):
        mu = mu_ind(t, exog)
        PV_ext[:, j] = stats.norm.cdf((mu - np.log(THRESH)) / np.sqrt(s2_ext)) * (1 - p_seqfail(t))
    W_ext = risk_w(ext_t)
    R_ext = np.tile(W_ext, (n, 1))
    N_ext = np.ones((n, len(ext_t)))
    for j in range(len(ext_t) - 1, -1, -1):
        if ext_t[j] > 30.0:
            continue
        k = j + step
        if k < len(ext_t):
            R_ext[:, j] = PV_ext[:, j] * W_ext[j] + (1 - PV_ext[:, j]) * (LAMBDA + R_ext[:, k])
            N_ext[:, j] = 1 + (1 - PV_ext[:, j]) * N_ext[:, k]
    jmap = np.round((T_GRID - ext_t[0]) / 0.25).astype(int)
    RM_ind = R_ext[:, jmap]
    NT_ind = N_ext[:, jmap]

    # ---------- 按 BMI 分组（DP，组内共用时点） ----------
    df['bmi_r'] = np.round(df['bmi'] / BMI_RES) * BMI_RES
    uniq = np.sort(df['bmi_r'].unique())
    # 每个唯一 BMI 值的平均个体风险曲线（含协变量异质性）
    RM_val = np.empty((len(uniq), len(T_GRID)))
    wcnt = np.zeros(len(uniq))
    for i, u in enumerate(uniq):
        idx = np.where(df['bmi_r'].values == u)[0]
        RM_val[i] = RM_ind[idx].mean(axis=0)
        wcnt[i] = len(idx)
    dp_out = {}
    for K in [4, 5, 6]:
        res = dp_partition(uniq, wcnt, K, RM_val, n_min=N_MIN)
        if res is None:
            continue
        bounds, tot, ts = res
        groups = []
        for (l, r), t in zip(bounds, ts):
            lo, hi = uniq[l], uniq[r]
            msk = (df['bmi_r'] >= lo - 1e-9) & (df['bmi_r'] <= hi + 1e-9)
            idx = np.where(msk.values)[0]
            Rg = RM_ind[idx].sum(axis=0)
            k2 = int(np.argmin(Rg))
            tstar = float(T_GRID[k2])
            # 达标比例：模型 + 经验（±1.5 周窗口）
            p_at = float(pmat[idx, k2].mean())
            near = df[msk & (df['gest'] >= tstar - 1.5) & (df['gest'] < tstar + 1.5)]
            emp_p = float((near['conc_y'] >= THRESH).mean()) if len(near) >= 5 else None
            # 个体化时点的改进
            t_ind = T_GRID[RM_ind[idx].argmin(axis=1)]
            r_ind = RM_ind[idx].min(axis=1).mean()
            groups.append({'bmi_interval': [round(float(lo), 1), round(float(hi), 1)],
                           'n': int(msk.sum()), 'best_t': tstar,
                           'mean_risk': round(float(Rg[k2] / len(idx)), 4),
                           'mean_ntests': round(float(NT_ind[idx, k2].mean()), 3),
                           'model_pass_rate': round(p_at, 3),
                           'emp_pass_rate': round(emp_p, 3) if emp_p is not None else None,
                           'n_emp_window': int(len(near)),
                           'ind_t_mean': round(float(t_ind.mean()), 2),
                           'ind_t_sd': round(float(t_ind.std()), 2),
                           'risk_if_individual': round(float(r_ind), 4)})
        dp_out[f'K{K}'] = {'total_risk_per_person': round(float(tot / n), 4), 'groups': groups}
    out['grouping'] = dp_out

    # Q2 模型对比（仅 gest+bmi+bmi²）下的总风险
    q2 = json.load(open(os.path.join(RES, 'q2_summary.json')))
    q2total = min(g['total_risk_per_person'] for g in q2['optimized_grouping'].values())
    out['compare'] = {'q2_total_risk': q2total,
                      'q3_total_risk': min(g['total_risk_per_person'] for g in dp_out.values())}

    # ---------- 检测误差敏感性（扩展模型） ----------
    bk = min(dp_out, key=lambda kk: dp_out[kk]['total_risk_per_person'])
    Kbest = int(bk[1])
    sens = []
    for cv in [0.0, 0.05, 0.10, 0.15, 0.20]:
        s2 = s2_ext + cv ** 2
        PV2 = np.empty((n, len(ext_t)))
        for j, t in enumerate(ext_t):
            mu = mu_ind(t, exog)
            PV2[:, j] = stats.norm.cdf((mu - np.log(THRESH)) / np.sqrt(s2)) * (1 - p_seqfail(t))
        R2 = np.tile(W_ext, (n, 1))
        for j in range(len(ext_t) - 1, -1, -1):
            if ext_t[j] > 30.0:
                continue
            k = j + step
            if k < len(ext_t):
                R2[:, j] = PV2[:, j] * W_ext[j] + (1 - PV2[:, j]) * (LAMBDA + R2[:, k])
        RM2 = R2[:, jmap]
        # 每个唯一 BMI 的平均曲线再 DP
        RMv2 = np.empty((len(uniq), len(T_GRID)))
        for i, u in enumerate(uniq):
            idx = np.where(df['bmi_r'].values == u)[0]
            RMv2[i] = RM2[idx].mean(axis=0)
        res = dp_partition(uniq, wcnt, Kbest, RMv2, n_min=N_MIN)
        rows = []
        if res:
            for (l, r), t in zip(res[0], res[2]):
                rows.append({'bmi_interval': [float(uniq[l]), float(uniq[r])],
                             'best_t': float(t)})
        sens.append({'cv_error': cv,
                     'total_risk': round(float(res[1] / n), 4) if res else None,
                     'group_best_t': rows})
    out['error_sensitivity'] = sens

    # ---------- 图 ----------
    # 图A：森林图——扩展模型系数（95% CI，标准化？用原始系数+CI）
    b = m.params
    ci = m.conf_int()
    names = [k for k in b.index if k != 'Intercept']
    est = [float(b[k]) for k in names]
    lo = [float(ci.loc[k, 0]) for k in names]
    hi = [float(ci.loc[k, 1]) for k in names]
    lab = {'gest': '孕周(周)', 'bmi': 'BMI', 'I(bmi ** 2)': 'BMI²',
           'weight': '体重(kg)', 'height': '身高(cm)', 'age': '年龄(岁)',
           'I(weight ** 2)': '体重²'}.get
    labels = [lab(k, k) for k in names]
    fig = forest(np.array(est), np.array(lo), np.array(hi), labels,
                 interval_label='95% CI（HC3 稳健）')
    fig.suptitle(f'扩展模型（{best}）ln(Y浓度) 回归系数及 95% CI', fontsize=12)
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q3_forest'))

    # 图B：分组与个体化时点
    bk = min(dp_out, key=lambda kk: dp_out[kk]['total_risk_per_person'])
    groups = dp_out[bk]['groups']
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x0 = []
    for i, g in enumerate(groups):
        lo, hi = g['bmi_interval']
        idx = np.where(((df['bmi_r'] >= lo - 1e-9) & (df['bmi_r'] <= hi + 1e-9)).values)[0]
        t_ind = T_GRID[RM_ind[idx].argmin(axis=1)]
        jitter = np.random.default_rng(7).uniform(-0.3, 0.3, len(idx))
        ax.scatter(df['bmi'].values[idx], t_ind + jitter, s=6, alpha=0.35, color='#4878CF')
        ax.hlines(g['best_t'], lo, hi, color='crimson', lw=3,
                  label=f"G{i+1} [{lo:.1f},{hi:.1f}] t*={g['best_t']:.1f}")
        x0.append((lo + hi) / 2)
    ax.set_xlabel('孕妇 BMI（kg/m²）'); ax.set_ylabel('最佳检测时点（孕周）')
    ax.set_title(f'个体化最优时点（散点）与组内统一时点（红线，{bk}）')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q3_timing'))

    # 图C：误差敏感性
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cvs = [s['cv_error'] * 100 for s in sens]
    tot = [s['total_risk'] for s in sens]
    ax.plot(cvs, tot, 's-', color='crimson')
    ax.set_xlabel('检测误差（浓度变异系数，%）')
    ax.set_ylabel('优化后人均期望风险')
    ax.set_title('扩展模型下检测误差对最小风险的影响')
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q3_error'))

    with open(os.path.join(RES, 'q3_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('OK q3; best =', best, '; grouping best =', bk)


if __name__ == '__main__':
    main()
