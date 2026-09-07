import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json

# Load male dataset
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
male['log_Y'] = np.log(male['Y染色体浓度'])

# 1. Descriptive stats
desc_stats = {
    'sample_size': len(male),
    'num_mothers': male['孕妇代码'].nunique(),
    'GA_mean': float(male['GA'].mean()),
    'GA_std': float(male['GA'].std()),
    'BMI_mean': float(male['孕妇BMI'].mean()),
    'BMI_std': float(male['孕妇BMI'].std()),
    'Y_conc_mean': float(male['Y染色体浓度'].mean()),
    'Y_conc_std': float(male['Y染色体浓度'].std()),
    'Y_conc_median': float(male['Y染色体浓度'].median()),
    'Y_conc_iqr': float(male['Y染色体浓度'].quantile(0.75) - male['Y染色体浓度'].quantile(0.25)),
    'pct_qualified': float((male['Y染色体浓度'] >= 0.04).mean() * 100)
}

# 2. Correlation analysis
corr_cols = [
    ('GA', '孕周 (GA)'),
    ('孕妇BMI', '孕妇BMI'),
    ('年龄', '孕妇年龄'),
    ('身高', '孕妇身高'),
    ('体重', '孕妇体重'),
    ('原始读段数', '原始读段数'),
    ('在参考基因组上比对的比例', '比对比例'),
    ('重复读段的比例', '重复读段比例'),
    ('唯一比对的读段数', '唯一比对读段数'),
    ('GC含量', 'GC含量'),
    ('被过滤掉读段数的比例', '过滤读段比例'),
    ('生产次数', '生产次数')
]

corr_rows = []
for col_key, col_label in corr_cols:
    pr, pp = stats.pearsonr(male[col_key], male['Y染色体浓度'])
    sr, sp = stats.spearmanr(male[col_key], male['Y染色体浓度'])
    corr_rows.append({
        'feature': col_label,
        'col_key': col_key,
        'pearson_r': pr,
        'pearson_p': pp,
        'spearman_rho': sr,
        'spearman_p': sp
    })

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv('analysis_output/q1_correlations.csv', index=False)

# 3. Regression Models
# Model 1: Simple linear
m1 = smf.ols('Y染色体浓度 ~ GA + 孕妇BMI', data=male).fit()

# Model 2: Interaction + Quadratic
m2 = smf.ols('Y染色体浓度 ~ GA + 孕妇BMI + I(GA**2) + I(孕妇BMI**2) + GA:孕妇BMI', data=male).fit()

# Model 3: Log-linear
m3 = smf.ols('log_Y ~ GA + 孕妇BMI', data=male).fit()

# Model 4: Multi-factor
m4 = smf.ols('Y染色体浓度 ~ GA + 孕妇BMI + 年龄 + 身高 + 原始读段数 + GC含量', data=male).fit()

# Model 5: Mixed linear model (repeated measures)
m5 = smf.mixedlm('Y染色体浓度 ~ GA + 孕妇BMI', data=male, groups=male['孕妇代码']).fit()

# Save model comparison summary
models_summary = {
    'M1_OLS_Linear': {
        'R2': float(m1.rsquared), 'Adj_R2': float(m1.rsquared_adj), 'AIC': float(m1.aic), 'BIC': float(m1.bic),
        'F_pvalue': float(m1.f_pvalue),
        'params': {k: float(v) for k, v in m1.params.items()},
        'pvalues': {k: float(v) for k, v in m1.pvalues.items()},
        'conf_int': {k: [float(v[0]), float(v[1])] for k, v in m1.conf_int().iterrows()}
    },
    'M2_Polynomial_Interaction': {
        'R2': float(m2.rsquared), 'Adj_R2': float(m2.rsquared_adj), 'AIC': float(m2.aic), 'BIC': float(m2.bic),
        'F_pvalue': float(m2.f_pvalue),
        'params': {k: float(v) for k, v in m2.params.items()},
        'pvalues': {k: float(v) for k, v in m2.pvalues.items()}
    },
    'M3_LogLinear': {
        'R2': float(m3.rsquared), 'Adj_R2': float(m3.rsquared_adj), 'AIC': float(m3.aic), 'BIC': float(m3.bic),
        'F_pvalue': float(m3.f_pvalue),
        'params': {k: float(v) for k, v in m3.params.items()},
        'pvalues': {k: float(v) for k, v in m3.pvalues.items()}
    },
    'M4_MultiFactor': {
        'R2': float(m4.rsquared), 'Adj_R2': float(m4.rsquared_adj), 'AIC': float(m4.aic), 'BIC': float(m4.bic),
        'F_pvalue': float(m4.f_pvalue),
        'params': {k: float(v) for k, v in m4.params.items()},
        'pvalues': {k: float(v) for k, v in m4.pvalues.items()}
    },
    'M5_LinearMixed': {
        'LogLik': float(m5.llf), 'AIC': float(m5.aic), 'BIC': float(m5.bic),
        'params': {k: float(v) for k, v in m5.params.items()},
        'pvalues': {k: float(v) for k, v in m5.pvalues.items()}
    }
}

with open('analysis_output/q1_models_summary.json', 'w') as f:
    json.dump(models_summary, f, indent=2)

print('Q1 analysis completed successfully!')
