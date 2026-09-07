import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar
import json

# 1. Load male data
male = pd.read_csv('test-arena-methodology/男胎检测数据.csv')
male.columns = [c.strip() for c in male.columns]

def parse_ga(val):
    if pd.isna(val): return np.nan
    val = str(val).strip()
    import re
    m = re.match(r'^(\d+)[wW](?:\+(\d+))?$', val)
    if m:
        w = int(m.group(1))
        d = int(m.group(2)) if m.group(2) else 0
        return w + d / 7.0
    return np.nan

male['GA'] = male['检测孕周'].apply(parse_ga)
male['qualified'] = (male['Y染色体浓度'] >= 0.04).astype(int)

# 2. Subject-level dataset
records = []
for code, g in male.groupby('孕妇代码'):
    g_sorted = g.sort_values('GA')
    bmi = g['孕妇BMI'].iloc[0]
    age = g['年龄'].iloc[0]
    ht = g['身高'].iloc[0]
    wt = g['体重'].iloc[0]
    par = g['生产次数'].iloc[0]
    gas = g_sorted['GA'].values
    ys = g_sorted['Y染色体浓度'].values
    
    pass_mask = ys >= 0.04
    if pass_mask.any():
        idx = np.where(pass_mask)[0][0]
        if idx == 0:
            t_pass = gas[0]
        else:
            y0, y1 = ys[idx-1], ys[idx]
            t0, t1 = gas[idx-1], gas[idx]
            t_pass = t0 + (0.04 - y0) / (y1 - y0) * (t1 - t0) if y1 != y0 else t1
        ever_pass = True
    else:
        t_pass = np.nan
        ever_pass = False
        
    records.append({
        'code': code, 'BMI': bmi, 'age': age, 'height': ht, 'weight': wt, 'parity': par,
        't_pass': t_pass, 'ever_pass': ever_pass, 'first_GA': gas[0], 'first_Y': ys[0],
        'n_tests': len(g), 'max_Y': ys.max()
    })

df_sub = pd.DataFrame(records)
df_sub.to_csv('analysis_output/male_subject_level.csv', index=False)

# 3. Fit models
ols_conc_q2 = smf.ols('Y染色体浓度 ~ GA + 孕妇BMI', data=male).fit()
ols_conc_q3 = smf.ols('Y染色体浓度 ~ GA + 孕妇BMI + 年龄 + 身高 + 生产次数', data=male).fit()
res_std_q2 = np.std(ols_conc_q2.resid)
res_std_q3 = np.std(ols_conc_q3.resid)

# Precompute vector coefficients
b0_q2, b_ga_q2, b_bmi_q2 = ols_conc_q2.params['Intercept'], ols_conc_q2.params['GA'], ols_conc_q2.params['孕妇BMI']

# Vectorized clinical loss
def clinical_loss_vec(t):
    t = np.asarray(t)
    loss = np.zeros_like(t, dtype=float)
    m1 = t <= 12
    m2 = (t > 12) & (t <= 27)
    m3 = t > 27
    loss[m1] = 1.0 * (t[m1] - 10)
    loss[m2] = 2.0 + 3.0 * (t[m2] - 12)
    loss[m3] = 2.0 + 3.0 * 15 + 10.0 * (t[m3] - 27)
    return loss

# Vectorized expected risk across a cohort at timing t
def group_risk_q2_vec(t_val, bmis, sigma_e=0.0, delta_t=2.5, c_fail=2.0):
    mu = b0_q2 + b_ga_q2 * t_val + b_bmi_q2 * bmis
    tot_sd = np.sqrt(res_std_q2**2 + sigma_e**2)
    p = 1.0 - stats.norm.cdf(0.04, loc=mu, scale=tot_sd)
    p = np.clip(p, 0.001, 0.999)
    l_succ = clinical_loss_vec(t_val)
    l_fail = clinical_loss_vec(t_val + delta_t) + c_fail
    r = p * l_succ + (1.0 - p) * l_fail
    return np.mean(r)

# Find optimal timing for any subset of bmis
def optimize_timing_for_bmis(bmis, sigma_e=0.0):
    t_grid = np.linspace(10.0, 22.0, 241)
    risks = [group_risk_q2_vec(t, bmis, sigma_e) for t in t_grid]
    best_idx = np.argmin(risks)
    best_t = t_grid[best_idx]
    best_r = risks[best_idx]
    # Compute average p_qual at best_t
    mu = b0_q2 + b_ga_q2 * best_t + b_bmi_q2 * bmis
    tot_sd = np.sqrt(res_std_q2**2 + sigma_e**2)
    p = 1.0 - stats.norm.cdf(0.04, loc=mu, scale=tot_sd)
    return best_t, best_r, np.mean(p)

# Evaluate a grouping scheme
def evaluate_bins_q2(bins, df, sigma_e=0.0):
    results = []
    tot_risk = 0.0
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        sub = df[(df['BMI'] >= low) & (df['BMI'] < high)]
        n = len(sub)
        if n == 0: continue
        bmis = sub['BMI'].values
        best_t, avg_r, avg_p = optimize_timing_for_bmis(bmis, sigma_e)
        tot_risk += avg_r * n
        sub_pass = sub['t_pass'].dropna()
        med_pass = sub_pass.median() if len(sub_pass) > 0 else np.nan
        results.append({
            'group': f'[{low:.1f}, {high:.1f})',
            'n_patients': int(n),
            'pct_patients': float(n / len(df) * 100),
            'mean_BMI': float(np.mean(bmis)),
            'opt_NIPT_timing': float(best_t),
            'avg_risk': float(avg_r),
            'avg_qual_prob': float(avg_p),
            'empirical_med_pass_GA': float(med_pass)
        })
    return results, tot_risk / len(df)

# 1. Standard Clinical Empirical Groups
emp_bins = [20.0, 28.0, 32.0, 36.0, 40.0, 50.0]
emp_res, emp_avg_r = evaluate_bins_q2(emp_bins, df_sub, sigma_e=0.0)

# 2. Data-Driven Quantile Groups (4 groups & 5 groups)
q4_cuts = [df_sub['BMI'].min() - 0.01] + list(np.quantile(df_sub['BMI'], [0.25, 0.50, 0.75])) + [df_sub['BMI'].max() + 0.01]
q4_res, q4_avg_r = evaluate_bins_q2(q4_cuts, df_sub, sigma_e=0.0)

q5_cuts = [df_sub['BMI'].min() - 0.01] + list(np.quantile(df_sub['BMI'], [0.20, 0.40, 0.60, 0.80])) + [df_sub['BMI'].max() + 0.01]
q5_res, q5_avg_r = evaluate_bins_q2(q5_cuts, df_sub, sigma_e=0.0)

# 3. Detection Error Sensitivity
sigma_e_grid = [0.0, 0.002, 0.005, 0.008, 0.010, 0.015, 0.020]
sens_q2 = []
for s_e in sigma_e_grid:
    _, r_emp = evaluate_bins_q2(emp_bins, df_sub, sigma_e=s_e)
    _, r_q4 = evaluate_bins_q2(q4_cuts, df_sub, sigma_e=s_e)
    t_med, _, _ = optimize_timing_for_bmis(np.array([31.8]), sigma_e=s_e)
    t_high, _, _ = optimize_timing_for_bmis(np.array([38.0]), sigma_e=s_e)
    sens_q2.append({
        'sigma_e': float(s_e),
        'emp_avg_risk': float(r_emp),
        'q4_avg_risk': float(r_q4),
        'opt_timing_BMI31_8': float(t_med),
        'opt_timing_BMI38': float(t_high)
    })

q2_data = {
    'empirical_grouping': {'bins': emp_bins, 'groups': emp_res, 'avg_population_risk': emp_avg_r},
    'optimal_4group': {'bins': [float(b) for b in q4_cuts], 'groups': q4_res, 'avg_population_risk': q4_avg_r},
    'optimal_5group': {'bins': [float(b) for b in q5_cuts], 'groups': q5_res, 'avg_population_risk': q5_avg_r},
    'sensitivity': sens_q2
}
with open('analysis_output/q2_output.json', 'w') as f:
    json.dump(q2_data, f, indent=2)

# ==================== Q3 Multi-Factor Vectorized ====================
b0_q3 = ols_conc_q3.params['Intercept']
b_ga_q3 = ols_conc_q3.params['GA']
b_bmi_q3 = ols_conc_q3.params['孕妇BMI']
b_age_q3 = ols_conc_q3.params['年龄']
b_ht_q3 = ols_conc_q3.params['身高']
b_par_q3 = ols_conc_q3.params['生产次数']

def group_risk_q3_vec(t_val, sub_df, sigma_e=0.0, delta_t=2.5, c_fail=2.0):
    mu = (b0_q3 + b_ga_q3 * t_val + 
          b_bmi_q3 * sub_df['BMI'].values + 
          b_age_q3 * sub_df['age'].values + 
          b_ht_q3 * sub_df['height'].values + 
          b_par_q3 * sub_df['parity'].values)
    tot_sd = np.sqrt(res_std_q3**2 + sigma_e**2)
    p = 1.0 - stats.norm.cdf(0.04, loc=mu, scale=tot_sd)
    p = np.clip(p, 0.001, 0.999)
    l_succ = clinical_loss_vec(t_val)
    l_fail = clinical_loss_vec(t_val + delta_t) + c_fail
    r = p * l_succ + (1.0 - p) * l_fail
    return np.mean(r), np.mean(p)

def evaluate_bins_q3(bins, df, sigma_e=0.0):
    results = []
    tot_risk = 0.0
    t_grid = np.linspace(10.0, 22.0, 241)
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        sub = df[(df['BMI'] >= low) & (df['BMI'] < high)]
        n = len(sub)
        if n == 0: continue
        
        risks_and_ps = [group_risk_q3_vec(t, sub, sigma_e) for t in t_grid]
        risks = [rp[0] for rp in risks_and_ps]
        ps = [rp[1] for rp in risks_and_ps]
        best_idx = np.argmin(risks)
        best_t = t_grid[best_idx]
        best_r = risks[best_idx]
        best_p = ps[best_idx]
        
        tot_risk += best_r * n
        results.append({
            'group': f'[{low:.1f}, {high:.1f})',
            'n_patients': int(n),
            'pct_patients': float(n / len(df) * 100),
            'mean_BMI': float(sub['BMI'].mean()),
            'mean_Age': float(sub['age'].mean()),
            'mean_Height': float(sub['height'].mean()),
            'mean_Weight': float(sub['weight'].mean()),
            'opt_NIPT_timing': float(best_t),
            'avg_risk': float(best_r),
            'avg_qual_prob': float(best_p)
        })
    return results, tot_risk / len(df)

q3_emp_res, q3_emp_r = evaluate_bins_q3(emp_bins, df_sub, sigma_e=0.0)
q3_q4_res, q3_q4_r = evaluate_bins_q3(q4_cuts, df_sub, sigma_e=0.0)
q3_q5_res, q3_q5_r = evaluate_bins_q3(q5_cuts, df_sub, sigma_e=0.0)

sens_q3 = []
for s_e in sigma_e_grid:
    _, r_emp = evaluate_bins_q3(emp_bins, df_sub, sigma_e=s_e)
    _, r_q4 = evaluate_bins_q3(q4_cuts, df_sub, sigma_e=s_e)
    sens_q3.append({
        'sigma_e': float(s_e),
        'emp_avg_risk': float(r_emp),
        'q4_avg_risk': float(r_q4)
    })

q3_data = {
    'empirical_grouping': {'bins': emp_bins, 'groups': q3_emp_res, 'avg_population_risk': q3_emp_r},
    'optimal_4group': {'bins': [float(b) for b in q4_cuts], 'groups': q3_q4_res, 'avg_population_risk': q3_q4_r},
    'optimal_5group': {'bins': [float(b) for b in q5_cuts], 'groups': q3_q5_res, 'avg_population_risk': q3_q5_r},
    'sensitivity': sens_q3
}
with open('analysis_output/q3_output.json', 'w') as f:
    json.dump(q3_data, f, indent=2)

print('Q2 and Q3 vectorized analyses completed successfully!')
