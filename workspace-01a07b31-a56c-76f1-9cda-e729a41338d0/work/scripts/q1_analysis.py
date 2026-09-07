# -*- coding: utf-8 -*-
"""问题1：Y 染色体浓度与孕周、BMI 的关系模型与显著性检验。"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(__file__))
from common import load_male, clean_male, WORK

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib_templates import evidence_mosaic, save_figure

import figure_style  # noqa

FIG = os.path.join(WORK, 'figures')
RES = os.path.join(WORK, 'results')
os.makedirs(RES, exist_ok=True)


def main():
    df, log = clean_male(load_male())
    df['lnC'] = np.log(df['conc_y'])
    df['lnG'] = np.log(df['gest'])
    df['lnBMI'] = np.log(df['bmi'])
    out = {'clean_log': log, 'n': len(df), 'n_mothers': int(df['mother'].nunique())}

    # ---------- 相关系数 ----------
    corr = {}
    for col in ['gest', 'bmi', 'weight', 'height', 'age', 'conc_x']:
        r_p, p_p = stats.pearsonr(df[col], df['conc_y'])
        r_s, p_s = stats.spearmanr(df[col], df['conc_y'])
        corr[col] = {'pearson_r': round(r_p, 4), 'pearson_p': f'{p_p:.2e}',
                     'spearman_r': round(r_s, 4), 'spearman_p': f'{p_s:.2e}'}
    out['correlations_vs_conc_y'] = corr

    # 同一孕妇重复测量下的组内相关系数（ICC）
    msb = df.groupby('mother')['conc_y'].agg(['mean', 'count'])
    grand_mean = df['conc_y'].mean()
    ssb = ((msb['mean'] - grand_mean) ** 2 * msb['count']).sum()
    ssw = df.groupby('mother')['conc_y'].apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
    n0 = (len(df) - (msb['count'] ** 2).sum() / len(df)) / (msb.shape[0] - 1)
    icc = (ssb / (len(df) - 1) - ssw / (len(df) - 1)) / (ssb / (len(df) - 1) + (n0 - 1) * ssw / (len(df) - 1)) \
        if ssw > 0 else np.nan
    out['icc_mother'] = round(float(icc), 4)

    # ---------- 候选回归模型 ----------
    formulas = {
        'M1_linear': 'conc_y ~ gest + bmi',
        'M2_loglin': 'lnC ~ gest + bmi',
        'M3_loglog': 'lnC ~ lnG + lnBMI',
        'M4_loglin_inter': 'lnC ~ gest + bmi + gest:bmi',
        'M5_quad': 'lnC ~ gest + bmi + I(gest**2) + I(bmi**2)',
        'M6_quad_bmi': 'lnC ~ gest + bmi + I(bmi**2)',
    }
    mods = {}
    for name, f in formulas.items():
        m = smf.ols(f, data=df).fit(cov_type='HC3')  # 异方差稳健标准误
        mods[name] = m
        out.setdefault('models', {})[name] = {
            'formula': f,
            'R2': round(m.rsquared, 4), 'R2_adj': round(m.rsquared_adj, 4),
            'AIC': round(m.aic, 1), 'BIC': round(m.bic, 1),
            'F_p': f'{m.f_pvalue:.2e}',
            'params': {k: {'coef': round(v, 5), 't': round(mods[name].tvalues[k], 2),
                           'p': f'{mods[name].pvalues[k]:.2e}'}
                       for k, v in m.params.items()},
        }

    # 不同响应变换的模型需用同一观测尺度（C）的对数似然比较：
    # lnC 正态模型对 C 的密度含雅可比项 1/C，llf_C = llf_lnC - sum(ln C_i)
    jac = np.log(df['conc_y'].values).sum()
    llC = {}
    for name, m in mods.items():
        llC[name] = m.llf if name == 'M1_linear' else m.llf - jac
    out['loglik_on_conc_scale'] = {k: round(v, 1) for k, v in llC.items()}
    out['model_tests'] = {
        'LR_M5_vs_M2': {'chi2': round(2 * (llC['M5_quad'] - llC['M2_loglin']), 2), 'df': 2,
                        'p': f'{stats.chi2.sf(2 * (llC["M5_quad"] - llC["M2_loglin"]), 2):.2e}'},
        'LR_M6_vs_M2': {'chi2': round(2 * (llC['M6_quad_bmi'] - llC['M2_loglin']), 2), 'df': 1,
                        'p': f'{stats.chi2.sf(2 * (llC["M6_quad_bmi"] - llC["M2_loglin"]), 1):.2e}'},
        'LR_M4_vs_M2': {'chi2': round(2 * (llC['M4_loglin_inter'] - llC['M2_loglin']), 2), 'df': 1,
                        'p': f'{stats.chi2.sf(2 * (llC["M4_loglin_inter"] - llC["M2_loglin"]), 1):.2e}'},
    }
    best_name = max(llC, key=lambda k: llC[k] - len(mods[k].params) * np.log(len(df)) / 2)
    out['model_selected_by_BIC'] = best_name

    # ---------- 混合效应模型（随机截距=孕妇） ----------
    mm = smf.mixedlm('lnC ~ gest + bmi', df, groups=df['mother']).fit(reml=False)
    out['mixed_model'] = {
        'params': {k: {'coef': round(v, 5), 'p': f'{mm.pvalues[k]:.2e}'} for k, v in mm.params.items()
                   if k in ['Intercept', 'gest', 'bmi']},
        'random_intercept_var': round(float(mm.cov_re.iloc[0, 0]), 6),
        'resid_var': round(float(mm.scale), 6),
        'llf': round(float(mm.llf), 1),
    }
    # 似然比：混合模型 vs 普通 OLS(M2)
    lr = 2 * (mm.llf - mods['M2_loglin'].llf)
    out['model_tests']['LR_mixed_vs_M2'] = {'chi2': round(lr, 2), 'p': f'{0.5 * stats.chi2.sf(lr, 1):.2e}'}

    # ---------- 残差诊断（最优对数线性模型） ----------
    m = mods['M2_loglin']
    resid = m.resid
    bp = sm.stats.diagnostic.het_breuschpagan(resid, m.model.exog)
    jb = stats.jarque_bera(resid)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = m.model.exog
    vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    out['diagnostics_M2'] = {
        'breusch_pagan': {'LM': round(bp[0], 2), 'p': f'{bp[1]:.2e}'},
        'jarque_bera': {'stat': round(jb.statistic, 2), 'p': f'{jb.pvalue:.2e}'},
        'VIF': {'intercept': round(vif[0], 2), 'gest': round(vif[1], 2), 'bmi': round(vif[2], 2)},
        'resid_sd': round(float(resid.std()), 5),
    }

    # ---------- 预测曲面：在 (孕周, BMI) 网格上的预测浓度 ----------
    g = np.linspace(10.5, 26, 63)
    b = np.linspace(20, 47, 55)
    G, B = np.meshgrid(g, b)
    beta = m.params
    C_pred = np.exp(beta['Intercept'] + beta['gest'] * G + beta['bmi'] * B)
    np.savez(os.path.join(RES, 'q1_surface.npz'), gest=g, bmi=b, conc=C_pred)

    # ---------- 图 ----------
    # 图1：散点 + 拟合带（孕周、BMI 各一幅，拼为 2 联）
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    ax.scatter(df['gest'], df['conc_y'] * 100, s=9, alpha=0.35, color='#4878CF', label='检测样本')
    bmid = np.median(df['bmi'])
    gg = np.linspace(df['gest'].min(), df['gest'].max(), 200)
    cc = np.exp(beta['Intercept'] + beta['gest'] * gg + beta['bmi'] * bmid) * 100
    ax.plot(gg, cc, 'r-', lw=2, label=f'模型预测（BMI={bmid:.1f}）')
    ax.axhline(4, color='k', ls='--', lw=1, label='4% 达标线')
    ax.set_xlabel('检测孕周（周）'); ax.set_ylabel('Y 染色体浓度（%）')
    ax.legend(frameon=False); ax.set_title('(a) 浓度-孕周')
    ax = axes[1]
    sc = ax.scatter(df['bmi'], df['conc_y'] * 100, c=df['gest'], s=9, alpha=0.45, cmap='viridis')
    fig.colorbar(sc, ax=ax, label='检测孕周（周）')
    bb = np.linspace(df['bmi'].min(), df['bmi'].max(), 200)
    gmid = np.median(df['gest'])
    cc2 = np.exp(beta['Intercept'] + beta['gest'] * gmid + beta['bmi'] * bb) * 100
    ax.plot(bb, cc2, 'r-', lw=2, label=f'模型预测（孕周={gmid:.1f}）')
    ax.axhline(4, color='k', ls='--', lw=1)
    ax.set_xlabel('孕妇 BMI（kg/m²）'); ax.set_ylabel('Y 染色体浓度（%）')
    ax.legend(frameon=False); ax.set_title('(b) 浓度-BMI')
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q1_scatter'))

    # 图2：预测曲面 + 4% 等值线
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    cs = ax.contourf(G, B, C_pred * 100, levels=20, cmap='viridis')
    fig.colorbar(cs, ax=ax, label='预测 Y 染色体浓度（%）')
    ax.contour(G, B, C_pred * 100, levels=[4.0], colors='r', linewidths=2)
    ax.clabel(ax.contour(G, B, C_pred * 100, levels=[4.0]), fmt='4%%')
    ax.plot(df['gest'], df['bmi'], 'k.', ms=1.5, alpha=0.25, label='样本')
    ax.set_xlabel('检测孕周（周）'); ax.set_ylabel('BMI（kg/m²）')
    ax.legend(loc='upper right', frameon=False)
    ax.set_title('模型预测的 Y 染色体浓度曲面（红线为 4% 达标线）')
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q1_surface'))

    # 图3：拟合诊断（evidence_mosaic 模板）
    pred = m.fittedvalues
    baseline = np.full(len(df), df['lnC'].mean())
    fig = evidence_mosaic(df['lnC'].values, df['lnC'].values, pred, baseline)
    fig.axes[0].set(xlabel='检测孕周（周）', ylabel='ln(Y 浓度)')
    fig.suptitle('对数线性模型拟合诊断（含基线比较与残差面板）', fontsize=12)
    save_figure(fig, os.path.join(FIG, 'fig_q1_diag'))

    # 图4：残差-拟合值
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(pred, resid, s=8, alpha=0.35, color='#6AACB8')
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel('拟合值 ln(Y 浓度)'); ax.set_ylabel('残差')
    ax.set_title('残差诊断（HC3 稳健标准误）')
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q1_residual'))

    with open(os.path.join(RES, 'q1_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])


if __name__ == '__main__':
    main()
