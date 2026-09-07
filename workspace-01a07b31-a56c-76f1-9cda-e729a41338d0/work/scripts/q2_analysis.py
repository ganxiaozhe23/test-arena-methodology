# -*- coding: utf-8 -*-
"""问题2：男胎孕妇 BMI 分组 + 最佳 NIPT 时点（潜在风险最小化）+ 检测误差影响。

模型与假设（均在论文中披露并做敏感性分析）：
  1) 浓度模型来自问题1：lnC(t,b)=β0+β1 t+β2 b+β3 b²，个体间方差 τ²、个体内方差 σ²。
  2) 单次检测"有效"需同时满足：测序成功（概率 1-p_fail(t)）且 Y 浓度 ≥4%。
     题目指出"检测时点过早"会导致测序失败，设
     p_fail(t)=p_base+p_early·exp(-κ(t-10))，基准 p_base=0.02, p_early=0.06, κ=0.5。
  3) 无效则间隔 δ=2 周复查；每次复查计附加代价 λ=0.05（抽血/等待成本）。
  4) 发现时点风险权重 w(D)：题目给定 <=12 周低、13-27 周高、>=28 周极高；
     取连续化分段线性：t<=12 → 1；12<t<=27 → 1+2(t-12)/15；t>27 → 3+3(t-27)。
  5) 期望风险 R(t0;b)=E[w(D)]+λ·E[复查次数]，反向递推计算。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from common import load_male, clean_male, WORK

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib_templates import raincloud, save_figure

import figure_style  # noqa
FIG = os.path.join(WORK, 'figures')
RES = os.path.join(WORK, 'results')
os.makedirs(RES, exist_ok=True)

THRESH = 0.04
T_GRID = np.arange(10, 25.01, 0.25)
DELTA = 2.0
LAMBDA = 0.05          # 每次复查附加代价
P_FAIL_BASE = 0.02
P_FAIL_EARLY = 0.06
KAPPA = 0.5
T_CAP = 30.0
BMI_RES = 0.5          # DP 分组时的 BMI 离散精度
N_MIN = 40             # 每组最少样本数


def load_q1_params():
    d = json.load(open(os.path.join(RES, 'q1_summary.json')))
    p = d['models']['M6_quad_bmi']['params']
    beta = np.array([p['Intercept']['coef'], p['gest']['coef'],
                     p['bmi']['coef'], p['I(bmi ** 2)']['coef']])
    return beta, d['mixed_model']['random_intercept_var'], \
        d['mixed_model']['resid_var'], d


def mu_ln(t, b, beta):
    return beta[0] + beta[1] * np.asarray(t) + beta[2] * np.asarray(b) + beta[3] * np.asarray(b) ** 2


def risk_w(t):
    t = np.asarray(t, dtype=float)
    return np.where(t <= 12, 1.0,
                    np.where(t <= 27, 1 + 2 * (t - 12) / 15.0,
                             3 + 3 * (t - 27)))


CFG = {'p_fail_early': P_FAIL_EARLY}


def p_seqfail(t):
    t = np.asarray(t, dtype=float)
    return P_FAIL_BASE + CFG['p_fail_early'] * np.exp(-KAPPA * (t - 10))


def pass_prob(t, b, beta, s2):
    return stats.norm.cdf((mu_ln(t, b, beta) - np.log(THRESH)) / np.sqrt(s2))


def valid_prob(t, b, beta, s2):
    return pass_prob(t, b, beta, s2) * (1 - p_seqfail(t))


def risk_curve_grid(b, beta, s2, delta=DELTA, lam=LAMBDA, grid=None,
                    vprob=None):
    """个体（或组代表）在 grid 各首次检测时点的期望风险，反向递推。"""
    if grid is None:
        grid = T_GRID
    tmax = max(grid.max(), T_CAP) + delta
    full = np.arange(round(grid.min(), 4), round(tmax, 4) + 1e-9, 0.25)
    p = valid_prob(full, b, beta, s2) if vprob is None else vprob(full, b)
    w = risk_w(full)
    R = np.empty(len(full))
    step = max(1, int(round(delta / 0.25)))
    for i in range(len(full) - 1, -1, -1):
        if full[i] > T_CAP:
            R[i] = w[i]
        else:
            j = i + step
            nxt = R[j] if j < len(full) else w[-1]
            R[i] = p[i] * w[i] + (1 - p[i]) * (lam + nxt)
    idx = np.clip(np.round((grid - full[0]) / 0.25).astype(int), 0, len(full) - 1)
    return R[idx]


def ntests_curve_grid(b, beta, s2, delta=DELTA, grid=None):
    """期望检测次数（含复查），同一反向递推。"""
    if grid is None:
        grid = T_GRID
    tmax = max(grid.max(), T_CAP) + delta
    full = np.arange(round(grid.min(), 4), round(tmax, 4) + 1e-9, 0.25)
    p = valid_prob(full, b, beta, s2)
    N = np.ones(len(full))
    step = max(1, int(round(delta / 0.25)))
    for i in range(len(full) - 1, -1, -1):
        if full[i] > T_CAP:
            N[i] = 1.0
        else:
            j = i + step
            nxt = N[j] if j < len(full) else 1.0
            N[i] = 1 + (1 - p[i]) * nxt
    idx = np.clip(np.round((grid - full[0]) / 0.25).astype(int), 0, len(full) - 1)
    return N[idx]


def risk_matrix(vals, beta, s2, delta=DELTA, lam=LAMBDA):
    M = np.empty((len(vals), len(T_GRID)))
    for i, v in enumerate(vals):
        M[i] = risk_curve_grid(v, beta, s2, delta, lam)
    return M


def dp_partition(vals, weights, K, RM, n_min=N_MIN):
    """区间划分 DP：最小化 Σ 样本期望风险，组内样本数 >= n_min。
    weights: 每个唯一 BMI 值的样本数。返回 (区间列表, 总风险, 每组时点)。"""
    n = len(vals)
    nt = RM.shape[1]
    wRM = RM * weights[:, None]
    pref = np.vstack([np.zeros(nt), np.cumsum(wRM, axis=0)])
    wcnt = np.concatenate([[0], np.cumsum(weights)])

    def seg_ok(l, r):
        return (wcnt[r + 1] - wcnt[l]) >= n_min

    def seg_cost(l, r):
        if not seg_ok(l, r):
            return np.inf, np.nan
        s = pref[r + 1] - pref[l]
        k = int(np.argmin(s))
        return s[k], T_GRID[k]

    INF = np.inf
    F = np.full((K + 1, n), INF)
    cut = np.full((K + 1, n), -1, dtype=int)
    for k in range(1, K + 1):
        for r in range(k - 1, n):
            if k == 1:
                c, _ = seg_cost(0, r)
                F[k, r] = c
                cut[k, r] = 0
                continue
            best, bj = INF, -1
            for j in range(k - 2, r):
                if F[k - 1, j] >= INF:
                    continue
                c0, _ = seg_cost(j + 1, r)
                if c0 >= INF:
                    continue
                c = F[k - 1, j] + c0
                if c < best:
                    best, bj = c, j
            if bj >= 0:
                F[k, r] = best
                cut[k, r] = bj + 1
    if cut[K, n - 1] < 0:
        return None
    bounds = []
    r = n - 1
    for k in range(K, 0, -1):
        l = cut[k, r]
        bounds.append((l, r))
        r = l - 1
    bounds.reverse()
    ts = [seg_cost(l, r)[1] for l, r in bounds]
    return bounds, F[K, n - 1], ts


def group_summary(df, vals, bounds, ts, beta, s2):
    groups = []
    for (l, r), t in zip(bounds, ts):
        lo, hi = vals[l], vals[r]
        msk = (df['bmi_r'] >= lo - 1e-9) & (df['bmi_r'] <= hi + 1e-9)
        bvals = df['bmi'][msk].values
        Rg = np.zeros(len(T_GRID))
        for bb in np.unique((np.round(bvals / BMI_RES) * BMI_RES)):
            Rg += risk_curve_grid(bb, beta, s2) * int(((np.round(bvals / BMI_RES) * BMI_RES) == bb).sum())
        k2 = int(np.argmin(Rg))
        Ng = np.zeros(len(T_GRID))
        for bb in np.unique(np.round(bvals / BMI_RES) * BMI_RES):
            Ng += ntests_curve_grid(bb, beta, s2) * int((np.round(bvals / BMI_RES) * BMI_RES == bb).sum())
        groups.append({'bmi_interval': [round(float(lo), 1), round(float(hi), 1)],
                       'n': int(msk.sum()), 'best_t': float(T_GRID[k2]),
                       'mean_risk': round(float(Rg[k2] / max(msk.sum(), 1)), 4),
                       'mean_ntests': round(float(Ng[k2] / max(msk.sum(), 1)), 3),
                       'pass_rate': round(float(np.mean(pass_prob(T_GRID[k2], bvals, beta, s2))), 3),
                       'valid_rate': round(float(np.mean(valid_prob(T_GRID[k2], bvals, beta, s2))), 3)})
    return groups


def main():
    df, log = clean_male(load_male())
    df['bmi_r'] = np.round(df['bmi'] / BMI_RES) * BMI_RES
    beta, tau2, sig2, q1 = load_q1_params()
    s2_base = tau2 + sig2
    out = {'n': len(df), 'delta': DELTA, 'lambda': LAMBDA,
           'p_fail': {'base': P_FAIL_BASE, 'early': P_FAIL_EARLY, 'kappa': KAPPA},
           'sigma2_total': round(s2_base, 4)}

    # ---------- 经验达标率验证 ----------
    df['pass'] = (df['conc_y'] >= THRESH).astype(int)
    gbin = pd.cut(df['gest'], bins=[10, 12, 14, 16, 18, 20, 22, 24, 26, 28], right=False)
    bbin = pd.cut(df['bmi'], bins=[20, 28, 32, 36, 40, 48], right=False)
    emp = df.groupby([gbin, bbin], observed=True)['pass'].agg(['mean', 'count'])
    emp_out = []
    for (g, b), row in emp.iterrows():
        if row['count'] >= 8:
            emp_out.append({'gest_bin': str(g), 'bmi_bin': str(b), 'n': int(row['count']),
                            'emp_rate': round(float(row['mean']), 3),
                            'model_rate': round(float(pass_prob(g.mid, b.mid, beta, s2_base)), 3)})
    out['empirical_validation'] = emp_out

    # ---------- 经验最早达标时间（左截断说明见论文） ----------
    emp_t = {}
    for bname, blo, bhi in [('20-28', 20, 28), ('28-32', 28, 32), ('32-36', 32, 36),
                            ('36-40', 36, 40), ('40+', 40, 99)]:
        sub = df[(df['bmi'] >= blo) & (df['bmi'] < bhi)]
        if len(sub) == 0:
            continue
        tstar = sub[sub['pass'] == 1].groupby('mother')['gest'].min()
        emp_t[bname] = {'n_mothers': int(sub['mother'].nunique()),
                        'n_reached': int(tstar.size),
                        'reach_rate': round(float(tstar.size / sub['mother'].nunique()), 3),
                        'median_first_pass_week': round(float(tstar.median()), 2) if tstar.size else None}
    out['empirical_first_pass_by_bmi'] = emp_t

    # 模型达标时间分位数：对每个 BMI，求 p(t,b)=q 的周数
    def t_quantile(b, q):
        ts = np.linspace(10, 26, 321)
        p = pass_prob(ts, b, beta, s2_base)
        hit = np.where(p >= q)[0]
        return float(ts[hit[0]]) if len(hit) else None
    out['model_time_to_reach'] = {}
    for bname, bm in [('BMI=25', 25), ('BMI=30', 30), ('BMI=35', 35),
                      ('BMI=40', 40), ('BMI=45', 45)]:
        out['model_time_to_reach'][bname] = {
            't_50%': t_quantile(bm, 0.5), 't_80%': t_quantile(bm, 0.8),
            't_95%': t_quantile(bm, 0.95)}

    # ---------- 经验分组 ----------
    uniq = np.unique((np.round(df['bmi'].values / BMI_RES) * BMI_RES))
    weights = np.array([((np.round(df['bmi'].values / BMI_RES) * BMI_RES) == u).sum() for u in uniq])
    RM = risk_matrix(uniq, beta, s2_base)
    row_idx = np.searchsorted(uniq, (np.round(df['bmi'].values / BMI_RES) * BMI_RES))
    emp_groups = [('20-28', 20, 28), ('28-32', 28, 32), ('32-36', 32, 36),
                  ('36-40', 36, 40), ('40+', 40, 99)]
    emp_res = []
    for name, lo, hi in emp_groups:
        msk = ((df['bmi'] >= lo) & (df['bmi'] < hi)).values
        if msk.sum() == 0:
            continue
        Rtot = RM[np.searchsorted(uniq, (np.round(df['bmi'][msk].values / BMI_RES) * BMI_RES))].sum(axis=0)
        k = int(np.argmin(Rtot))
        bvals = df['bmi'][msk].values
        emp_res.append({'group': name, 'n': int(msk.sum()), 'best_t': float(T_GRID[k]),
                        'mean_risk': round(float(Rtot[k] / msk.sum()), 4),
                        'pass_rate_at_best_t': round(float(np.mean(
                            pass_prob(T_GRID[k], bvals, beta, s2_base))), 3)})
    out['empirical_grouping_result'] = emp_res
    out['empirical_total_risk'] = round(
        float(RM[row_idx].min(axis=1).mean()), 4)  # 个体各自最优时的下界参考
    out['empirical_group_total_risk'] = round(
        sum(r['mean_risk'] * r['n'] for r in emp_res) / len(df), 4)

    # ---------- 数据驱动分组（DP + 最小样本量约束） ----------
    dp_out = {}
    for K in [4, 5, 6]:
        res = dp_partition(uniq, weights, K, RM)
        if res is None:
            continue
        bounds, tot, ts = res
        groups = group_summary(df, uniq, bounds, ts, beta, s2_base)
        dp_out[f'K{K}'] = {'total_risk_per_person': round(float(tot / len(df)), 4),
                           'groups': groups}
    out['optimized_grouping'] = dp_out

    # ---------- Bootstrap 切点不确定性（K 固定为最优） ----------
    bestK = min(dp_out, key=lambda kk: dp_out[kk]['total_risk_per_person'])
    rng = np.random.default_rng(42)
    n_boot = 200
    boot_cuts = []
    mothers = df['mother'].unique()
    for _ in range(n_boot):
        sel_m = rng.choice(mothers, size=len(mothers), replace=True)
        # 重抽孕妇（聚类 bootstrap），展开其全部检测记录
        parts = [df[df['mother'] == m] for m in sel_m]
        db = pd.concat(parts)
        wb = np.array([((np.round(db['bmi'].values / BMI_RES) * BMI_RES) == u).sum() for u in uniq])
        res = dp_partition(uniq, wb, int(bestK[1]), RM)
        if res is None:
            continue
        bounds, _, _ = res
        boot_cuts.append([uniq[l] for l, r in bounds][1:] + [uniq[bounds[-1][1]]])
    boot_cuts = np.array(boot_cuts)
    out['bootstrap'] = {'n_boot': len(boot_cuts), 'K': bestK,
                        'cut_ci_low': [round(float(x), 1) for x in np.percentile(boot_cuts, 2.5, axis=0)],
                        'cut_ci_high': [round(float(x), 1) for x in np.percentile(boot_cuts, 97.5, axis=0)],
                        'cut_median': [round(float(x), 1) for x in np.percentile(boot_cuts, 50, axis=0)]}

    # ---------- 检测误差敏感性 ----------
    sens = []
    for cv in [0.0, 0.05, 0.10, 0.15, 0.20]:
        s2 = s2_base + cv ** 2
        RM_s = risk_matrix(uniq, beta, s2)
        rows = []
        for name, lo, hi in emp_groups:
            msk = ((df['bmi'] >= lo) & (df['bmi'] < hi)).values
            if msk.sum() == 0:
                continue
            Rtot = RM_s[np.searchsorted(uniq, (np.round(df['bmi'][msk].values / BMI_RES) * BMI_RES))].sum(axis=0)
            k = int(np.argmin(Rtot))
            rows.append({'group': name, 'best_t': float(T_GRID[k]),
                         'mean_risk': round(float(Rtot[k] / msk.sum()), 4)})
        # 优化分组在误差下的总风险
        res = dp_partition(uniq, weights, int(bestK[1]), RM_s)
        tot_r = res[1] / len(df) if res else None
        sens.append({'cv_error': cv, 'rows': rows,
                     'optimized_total_risk': round(float(tot_r), 4) if tot_r else None})
    out['error_sensitivity'] = sens

    # 测序失败假设的敏感性
    fs = []
    for pe in [0.0, 0.06, 0.12, 0.20]:
        CFG['p_fail_early'] = pe
        RM_f = risk_matrix(uniq, beta, s2_base)
        res = dp_partition(uniq, weights, int(bestK[1]), RM_f)
        fs.append({'p_fail_early': pe,
                   'optimized_total_risk': round(float(res[1] / len(df)), 4) if res else None,
                   'best_t_by_group': [float(t) for t in res[2]] if res else None})
    CFG['p_fail_early'] = 0.06
    out['fail_sensitivity'] = fs

    # 复查间隔敏感性
    ds = []
    for dlt in [1.0, 2.0, 3.0]:
        RM_d = risk_matrix(uniq, beta, s2_base, delta=dlt)
        res = dp_partition(uniq, weights, int(bestK[1]), RM_d)
        ds.append({'delta': dlt,
                   'optimized_total_risk': round(float(res[1] / len(df)), 4) if res else None,
                   'best_t_by_group': [float(t) for t in res[2]] if res else None})
    out['delta_sensitivity'] = ds

    # ---------- 图 ----------
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(emp_groups)))
    # 图A：达标/有效概率曲线 + 经验点
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    tt = np.linspace(10, 26, 130)
    for (name, lo, hi), c in zip(emp_groups, colors):
        bmid = float(np.mean([lo, min(hi, 47)]))
        ax.plot(tt, pass_prob(tt, bmid, beta, s2_base) * 100, color=c, lw=2,
                label=f'BMI {name}')
        sub = df[(df['bmi'] >= lo) & (df['bmi'] < hi)]
        gb = pd.cut(sub['gest'], bins=[10, 13, 16, 19, 22, 25, 28], right=False)
        e = sub.groupby(gb, observed=True)['pass'].agg(['mean', 'count'])
        e = e[e['count'] >= 6]
        ax.plot([iv.mid for iv in e.index], e['mean'] * 100, 'o', color=c, ms=5, alpha=0.8)
    ax.axhline(95, color='gray', ls=':', lw=1)
    ax.text(10.2, 96, '95% 概率', color='gray', fontsize=8)
    ax.set_xlabel('检测孕周（周）'); ax.set_ylabel('Y 浓度 ≥4% 的概率（%）')
    ax.set_title('各 BMI 组浓度达标概率：模型曲线与经验比例（点）')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q2_passprob'))

    # 图B：期望风险曲线与最优点
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for (name, lo, hi), c in zip(emp_groups, colors):
        msk = ((df['bmi'] >= lo) & (df['bmi'] < hi)).values
        if msk.sum() == 0:
            continue
        Rtot = RM[np.searchsorted(uniq, (np.round(df['bmi'][msk].values / BMI_RES) * BMI_RES))].sum(axis=0) / msk.sum()
        k = int(np.argmin(Rtot))
        ax.plot(T_GRID, Rtot, color=c, lw=2, label=f'BMI {name}')
        ax.plot(T_GRID[k], Rtot[k], 'o', color=c, ms=7, mfc='white', mew=2)
        ax.annotate(f'{T_GRID[k]:.1f}周', (T_GRID[k], Rtot[k]), textcoords='offset points',
                    xytext=(6, -14), fontsize=8, color=c)
    ax.set_xlabel('首次检测时点（孕周）'); ax.set_ylabel('期望风险 E[w(D)]+λ·E[复查次数]')
    ax.set_title('不同首次检测时点的期望潜在风险（空心点为组内最优）')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q2_risk'))

    # 图C：优化分组雨云图
    groups = dp_out[bestK]['groups']
    data_g, labels = [], []
    for i, g in enumerate(groups):
        lo, hi = g['bmi_interval']
        v = df['bmi'][(df['bmi_r'] >= lo - 1e-9) & (df['bmi_r'] <= hi + 1e-9)].values
        data_g.append(v)
        labels.append(f"G{i+1}: {lo:.1f}-{hi:.1f}\nn={g['n']}, t*={g['best_t']:.1f}周")
    fig = raincloud(data_g, labels)
    fig.suptitle(f'数据驱动 {bestK} 分组 BMI 分布及各组最佳检测时点', fontsize=12)
    fig.axes[0].set(xlabel='孕妇 BMI（kg/m²）')
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q2_groups'))

    # 图D：误差敏感性
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    cvs = np.array([s['cv_error'] for s in sens]) * 100
    ax = axes[0]
    for gname in [g[0] for g in emp_groups]:
        ts = [next(r['best_t'] for r in s['rows'] if r['group'] == gname) for s in sens]
        ax.plot(cvs, ts, 'o-', label=gname)
    ax.set_xlabel('检测误差（浓度变异系数，%）'); ax.set_ylabel('最优检测时点（周）')
    ax.set_title('各组最优时点随检测误差的变化'); ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    tot_r = [s['optimized_total_risk'] for s in sens]
    ax.plot(cvs, tot_r, 's-', color='crimson', label='优化分组总风险')
    for gname in [g[0] for g in emp_groups]:
        rs = [next(r['mean_risk'] for r in s['rows'] if r['group'] == gname) for s in sens]
        ax.plot(cvs, rs, 'o-', label=gname, alpha=0.8)
    ax.set_xlabel('检测误差（浓度变异系数，%）'); ax.set_ylabel('最小期望风险')
    ax.set_title('最小期望风险随检测误差的变化'); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save_figure(fig, os.path.join(FIG, 'fig_q2_error'))

    with open(os.path.join(RES, 'q2_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('OK q2; bestK =', bestK)


if __name__ == '__main__':
    main()
