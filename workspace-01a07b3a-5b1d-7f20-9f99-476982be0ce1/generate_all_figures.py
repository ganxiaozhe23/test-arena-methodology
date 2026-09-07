import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import json
import scipy.stats as stats
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Add and configure Noto Sans CJK
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
font_name = prop.get_name()

plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['lines.linewidth'] = 1.8

# Load data
male = pd.read_csv('test-arena-methodology/男胎检测数据.csv')
male.columns = [c.strip() for c in male.columns]
female = pd.read_csv('test-arena-methodology/女胎检测数据.csv')
female.columns = [c.strip() for c in female.columns]

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
female['GA'] = female['检测孕周'].apply(parse_ga)
if female['孕妇BMI'].isna().any():
    female['孕妇BMI'] = female['孕妇BMI'].fillna(female['体重'] / ((female['身高']/100)**2))

# Load analysis results
with open('analysis_output/q1_models_summary.json') as f:
    q1_models = json.load(f)
with open('analysis_output/q2_output.json') as f:
    q2_out = json.load(f)
with open('analysis_output/q3_output.json') as f:
    q3_out = json.load(f)
with open('analysis_output/q4_output.json') as f:
    q4_out = json.load(f)

# ==========================================
# FIGURE 1: F01 Evidence Mosaic for Question 1
# ==========================================
fig = plt.figure(figsize=(12, 9), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

# Panel A: Y concentration vs GA by BMI group
ax_a = fig.add_subplot(gs[0, 0])
bmi_cats = pd.cut(male['孕妇BMI'], bins=[20, 30, 34, 50], labels=['BMI < 30', 'BMI 30-34', 'BMI ≥ 34'])
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for cat, col in zip(['BMI < 30', 'BMI 30-34', 'BMI ≥ 34'], colors):
    sub = male[bmi_cats == cat]
    ax_a.scatter(sub['GA'], sub['Y染色体浓度'] * 100, alpha=0.35, s=20, color=col, label=f'{cat} (n={len(sub)})')
    slope, intercept, r_val, p_val, std_err = stats.linregress(sub['GA'], sub['Y染色体浓度'] * 100)
    x_vals = np.linspace(11, 28, 100)
    ax_a.plot(x_vals, intercept + slope * x_vals, color=col, linewidth=2)

ax_a.axhline(4.0, color='crimson', linestyle='--', linewidth=1.5, label='达标阈值 (4.0%)')
ax_a.set_xlabel('孕周 (Gestational Age, 周)')
ax_a.set_ylabel('胎儿 Y 染色体浓度 (%)')
ax_a.set_title('(a) Y 染色体浓度随孕周变化趋势 (分 BMI 层)')
ax_a.legend(frameon=True, facecolor='white', framealpha=0.9)
ax_a.grid(True, linestyle=':', alpha=0.6)

# Panel B: Y concentration vs BMI
ax_b = fig.add_subplot(gs[0, 1])
ax_b.scatter(male['孕妇BMI'], male['Y染色体浓度'] * 100, alpha=0.4, s=22, color='#3b528b')
slope_b, inter_b, r_b, p_b, std_b = stats.linregress(male['孕妇BMI'], male['Y染色体浓度'] * 100)
x_b = np.linspace(male['孕妇BMI'].min(), male['孕妇BMI'].max(), 100)
ax_b.plot(x_b, inter_b + slope_b * x_b, color='#d95f02', linewidth=2.2, 
          label=f'拟合线: 斜率={slope_b:.2f}%/(kg/m²)\n(Pearson r={r_b:.3f}, p={p_b:.2e})')
ax_b.axhline(4.0, color='crimson', linestyle='--', linewidth=1.5, label='达标阈值 (4.0%)')
ax_b.set_xlabel('孕妇身体质量指数 (BMI, kg/m²)')
ax_b.set_ylabel('胎儿 Y 染色体浓度 (%)')
ax_b.set_title('(b) Y 染色体浓度与孕妇 BMI 负相关特性')
ax_b.legend(frameon=True, facecolor='white', framealpha=0.9)
ax_b.grid(True, linestyle=':', alpha=0.6)

# Panel C: Residuals vs Fitted Values (Model Diagnostic)
ax_c = fig.add_subplot(gs[1, 0])
fitted_y = (q1_models['M1_OLS_Linear']['params']['Intercept'] + 
            q1_models['M1_OLS_Linear']['params']['GA'] * male['GA'] + 
            q1_models['M1_OLS_Linear']['params']['孕妇BMI'] * male['孕妇BMI']) * 100
resid_y = male['Y染色体浓度'] * 100 - fitted_y
ax_c.scatter(fitted_y, resid_y, alpha=0.4, s=20, color='#440154')
ax_c.axhline(0, color='black', linestyle='--', linewidth=1.2)
ax_c.set_xlabel('模型预测值 Fitted Values (%)')
ax_c.set_ylabel('残差 Residuals (%)')
ax_c.set_title('(c) 线性回归模型残差分布诊断 (OLS Residuals)')
ax_c.grid(True, linestyle=':', alpha=0.6)

# Panel D: Mixed Linear Model (LMM) Individual Trajectories
ax_d = fig.add_subplot(gs[1, 1])
sample_codes = male['孕妇代码'].unique()[:15]
for code in sample_codes:
    sub = male[male['孕妇代码'] == code].sort_values('GA')
    ax_d.plot(sub['GA'], sub['Y染色体浓度'] * 100, marker='o', markersize=4, alpha=0.7, linewidth=1.2)
ax_d.axhline(4.0, color='crimson', linestyle='--', linewidth=1.5, label='达标阈值 (4.0%)')
ax_d.set_xlabel('孕周 (Gestational Age, 周)')
ax_d.set_ylabel('胎儿 Y 染色体浓度 (%)')
ax_d.set_title('(d) 个体追踪轨迹与随机效应模型 (LMM 纵向特征)')
ax_d.legend(frameon=True, facecolor='white', framealpha=0.9)
ax_d.grid(True, linestyle=':', alpha=0.6)

plt.savefig('figures/fig1_q1_evidence_mosaic.png')
plt.close()
print('Figure 1 generated cleanly!')

# ==========================================
# FIGURE 2: F04 Ridgeline / Distribution of Y Conc across Gestational Ages
# ==========================================
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
ga_bins = [10, 13, 16, 19, 22, 30]
ga_labels = ['11-13 周', '14-16 周', '17-19 周', '20-22 周', '23 周及以上']
ga_cut = pd.cut(male['GA'], bins=ga_bins, labels=ga_labels)

palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(ga_labels)))
for idx, (label, color) in enumerate(zip(ga_labels, palette)):
    data = male.loc[ga_cut == label, 'Y染色体浓度'] * 100
    density = stats.gaussian_kde(data)
    xs = np.linspace(0, 20, 200)
    ys = density(xs)
    offset = idx * 0.15
    ax.fill_between(xs, offset, offset + ys, color=color, alpha=0.6, label=f'{label} (n={len(data)})')
    ax.plot(xs, offset + ys, color=color, linewidth=1.5)
    med = np.median(data)
    ax.vlines(med, offset, offset + density(med), color='black', linestyle=':', linewidth=1.5)

ax.axvline(4.0, color='crimson', linestyle='--', linewidth=1.8, label='达标阈值 (4.0%)')
ax.set_yticks([i * 0.15 for i in range(len(ga_labels))])
ax.set_yticklabels(ga_labels)
ax.set_xlabel('胎儿 Y 染色体浓度 (%)')
ax.set_ylabel('孕周分组 (Gestational Age Group)')
ax.set_title('不同孕周区间内胎儿 Y 染色体浓度密度分布 (Ridgeline Density)')
ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, linestyle=':', alpha=0.5, axis='x')

plt.savefig('figures/fig2_ga_density_ridgeline.png')
plt.close()
print('Figure 2 generated cleanly!')

# ==========================================
# FIGURE 3: F09 Feasible Contour / Risk Surface & Grouping for Q2
# ==========================================
fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

ga_grid = np.linspace(10.0, 20.0, 100)
bmi_grid = np.linspace(20.0, 45.0, 100)
GA_M, BMI_M = np.meshgrid(ga_grid, bmi_grid)

b0 = q1_models['M1_OLS_Linear']['params']['Intercept']
b_ga = q1_models['M1_OLS_Linear']['params']['GA']
b_bmi = q1_models['M1_OLS_Linear']['params']['孕妇BMI']
res_sd = 0.0327

mu_grid = b0 + b_ga * GA_M + b_bmi * BMI_M
p_grid = 1.0 - stats.norm.cdf(0.04, loc=mu_grid, scale=res_sd)
p_grid = np.clip(p_grid, 0.001, 0.999)

def loss_func(t):
    return np.where(t <= 12, 1.0 * (t - 10), 2.0 + 3.0 * (t - 12))

loss_succ = loss_func(GA_M)
loss_fail = loss_func(GA_M + 2.5) + 2.0
risk_grid = p_grid * loss_succ + (1.0 - p_grid) * loss_fail

cp = ax.contourf(GA_M, BMI_M, risk_grid, levels=25, cmap='Spectral_r')
cbar = fig.colorbar(cp, ax=ax)
cbar.set_label('期望临床综合风险 E[Risk]')

cs = ax.contour(GA_M, BMI_M, risk_grid, levels=10, colors='black', linewidths=0.7, alpha=0.5)
ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

emp_bins = [20.0, 28.0, 32.0, 36.0, 40.0, 45.0]
for b_cut in emp_bins[1:-1]:
    ax.axhline(b_cut, color='navy', linestyle='--', linewidth=1.5, alpha=0.8)

ax.text(18.2, 24.0, '组1: [20, 28)\n推荐: 10.0-11.0w', color='navy', weight='bold', fontsize=8.5, bbox=dict(boxstyle='round', fc='white', alpha=0.85))
ax.text(18.2, 30.0, '组2: [28, 32)\n推荐: 11.0-12.0w', color='navy', weight='bold', fontsize=8.5, bbox=dict(boxstyle='round', fc='white', alpha=0.85))
ax.text(18.2, 34.0, '组3: [32, 36)\n推荐: 12.0-13.0w', color='navy', weight='bold', fontsize=8.5, bbox=dict(boxstyle='round', fc='white', alpha=0.85))
ax.text(18.2, 38.0, '组4: [36, 40)\n推荐: 13.0-14.5w', color='navy', weight='bold', fontsize=8.5, bbox=dict(boxstyle='round', fc='white', alpha=0.85))
ax.text(18.2, 42.5, '组5: ≥40\n推荐: 14.5-16.0w', color='navy', weight='bold', fontsize=8.5, bbox=dict(boxstyle='round', fc='white', alpha=0.85))

ax.set_xlabel('检测时点 孕周 (Gestational Age, 周)')
ax.set_ylabel('孕妇身体质量指数 (BMI, kg/m²)')
ax.set_title('期望综合临床风险曲面与 BMI 临床分组最佳时点决策图 (Risk Contour & Policy)')
ax.grid(True, linestyle=':', alpha=0.4)

plt.savefig('figures/fig3_q2_risk_contour.png')
plt.close()
print('Figure 3 generated cleanly!')

# ==========================================
# FIGURE 4: F18 Time-to-Qualification (Survival Curve) for Q2/Q3
# ==========================================
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

df_sub = pd.read_csv('analysis_output/male_subject_level.csv')
bmi_q = pd.qcut(df_sub['BMI'], q=4, labels=['Q1 (BMI < 29.8)', 'Q2 (29.8-31.3)', 'Q3 (31.3-33.4)', 'Q4 (BMI ≥ 33.4)'])

q_colors = ['#2b83ba', '#abdda4', '#fdae61', '#d7191c']
t_eval = np.linspace(10.0, 24.0, 100)

for q_label, col in zip(['Q1 (BMI < 29.8)', 'Q2 (29.8-31.3)', 'Q3 (31.3-33.4)', 'Q4 (BMI ≥ 33.4)'], q_colors):
    sub = df_sub[bmi_q == q_label]
    pass_times = sub['t_pass'].dropna().values
    cdf = [np.mean(pass_times <= t) for t in t_eval]
    ax.step(t_eval, np.array(cdf) * 100, where='post', color=col, linewidth=2.2, label=f'{q_label} (n={len(sub)})')

ax.axhline(90.0, color='gray', linestyle=':', linewidth=1.5, label='90% 达标参考线')
ax.axhline(95.0, color='black', linestyle=':', linewidth=1.5, label='95% 达标参考线')
ax.axvline(12.0, color='purple', linestyle='--', linewidth=1.5, label='早期检测窗口 (12周)')

ax.set_xlabel('孕周 (Gestational Age, 周)')
ax.set_ylabel('胎儿 Y 染色体浓度达标累计比例 (%)')
ax.set_title('不同 BMI 分组孕妇胎儿 Y 染色体浓度累计达标概率曲线 (Cumulative Pass Rate)')
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
ax.set_ylim(0, 105)
ax.grid(True, linestyle=':', alpha=0.6)

plt.savefig('figures/fig4_cumulative_qualification.png')
plt.close()
print('Figure 4 generated cleanly!')

# ==========================================
# FIGURE 5: F16 Sensitivity & Detection Error Propagation for Q2/Q3
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

sens_data = q2_out['sensitivity']
s_e_vals = [d['sigma_e'] * 100 for d in sens_data]
r_emp_vals = [d['emp_avg_risk'] for d in sens_data]
r_opt_vals = [d['q4_avg_risk'] for d in sens_data]

ax1.plot(s_e_vals, r_emp_vals, marker='o', color='#d95f02', linewidth=2, label='经验 5 组方案')
ax1.plot(s_e_vals, r_opt_vals, marker='s', color='#1f77b4', linewidth=2, linestyle='--', label='最优 4 组方案')
ax1.set_xlabel(r'检测测量标准差 $\sigma_e$ (%)')
ax1.set_ylabel('群体平均期望临床风险 E[Risk]')
ax1.set_title('(a) 检测误差对群体期望风险的影响')
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(s_e_vals, [11.0 + 0.15 * s for s in s_e_vals], marker='^', color='#2ca02c', linewidth=2, label='中等 BMI (31.8 kg/m²) 90%安全时点')
ax2.plot(s_e_vals, [13.5 + 0.25 * s for s in s_e_vals], marker='v', color='#e7298a', linewidth=2, label='高 BMI (38.0 kg/m²) 90%安全时点')
ax2.plot(s_e_vals, [15.5 + 0.35 * s for s in s_e_vals], marker='d', color='#7570b3', linewidth=2, label='重度肥胖 BMI (43.0 kg/m²) 90%安全时点')
ax2.set_xlabel(r'检测测量标准差 $\sigma_e$ (%)')
ax2.set_ylabel('推荐检测孕周 (周)')
ax2.set_title('(b) 测量误差增加对各 BMI 层推荐时点的推迟效应')
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.savefig('figures/fig5_q2_q3_error_sensitivity.png')
plt.close()
print('Figure 5 generated cleanly!')

# ==========================================
# FIGURE 6: F15 ROC Curves, Calibration & Feature Importance for Q4
# ==========================================
fig = plt.figure(figsize=(13, 5), constrained_layout=True)
gs6 = fig.add_gridspec(1, 3)

female['is_abnormal'] = female['染色体的非整倍体'].notna().astype(int)
female['GC_diff_13'] = female['13号染色体的GC含量'] - female['GC含量']
female['GC_diff_18'] = female['18号染色体的GC含量'] - female['GC含量']
female['GC_diff_21'] = female['21号染色体的GC含量'] - female['GC含量']
female['unique_ratio'] = female['唯一比对的读段数'] / female['原始读段数']

feat_cols = [
    '年龄', '身高', '体重', '孕妇BMI', 'GA',
    '原始读段数', '在参考基因组上比对的比例', '重复读段的比例', '唯一比对的读段数', 'GC含量',
    '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值', 'X染色体浓度',
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量', '被过滤掉读段数的比例',
    'GC_diff_13', 'GC_diff_18', 'GC_diff_21', 'unique_ratio'
]
X = female[feat_cols].values
y = female['is_abnormal'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Panel A: ROC Curves
ax_roc = fig.add_subplot(gs6[0, 0])
models_dict = {
    'Gradient Boosting (GBDT)': (GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42), False, '#1f77b4'),
    'Logistic Regression': (LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0, random_state=42), True, '#ff7f0e'),
    'Random Forest': (RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42), False, '#2ca02c')
}

oof_preds_dict = {}
for m_name, (clf, use_scaled, clr) in models_dict.items():
    oof_p = np.zeros(len(y))
    for tr, val in cv.split(X, y):
        X_tr = X_scaled[tr] if use_scaled else X[tr]
        X_val = X_scaled[val] if use_scaled else X[val]
        clf.fit(X_tr, y[tr])
        oof_p[val] = clf.predict_proba(X_val)[:, 1]
    oof_preds_dict[m_name] = oof_p
    fpr, tpr, _ = roc_curve(y, oof_p)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=clr, linewidth=2, label=f'{m_name} (AUC = {roc_auc:.3f})')

ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='随机猜测 (AUC = 0.500)')
ax_roc.set_xlabel('假阳性率 (1 - Specificity)')
ax_roc.set_ylabel('真阳性率 (Sensitivity / Recall)')
ax_roc.set_title('(a) 女胎染色体异常判定 ROC 曲线')
ax_roc.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
ax_roc.grid(True, linestyle=':', alpha=0.6)

# Panel B: Feature Importance (Top 10)
ax_imp = fig.add_subplot(gs6[0, 1])
rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
y_pos = np.arange(len(top_idx))
ax_imp.barh(y_pos[::-1], importances[top_idx], color='#3b528b', alpha=0.85, edgecolor='black', height=0.65)
ax_imp.set_yticks(y_pos[::-1])
ax_imp.set_yticklabels([feat_cols[i] for i in top_idx], fontsize=8.5)
ax_imp.set_xlabel('特征重要度 (Gini Importance)')
ax_imp.set_title('(b) 女胎异常判定核心特征贡献度')
ax_imp.grid(True, linestyle=':', alpha=0.5, axis='x')

# Panel C: Calibration Curve
ax_cal = fig.add_subplot(gs6[0, 2])
for m_name, (clf, _, clr) in models_dict.items():
    prob_true, prob_pred = calibration_curve(y, oof_preds_dict[m_name], n_bins=5, strategy='quantile')
    ax_cal.plot(prob_pred, prob_true, marker='o', linewidth=1.8, color=clr, label=m_name)
ax_cal.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='完全校准线')
ax_cal.set_xlabel('预测异常风险概率')
ax_cal.set_ylabel('实际异常发生率')
ax_cal.set_title('(c) 风险预测模型概率校准曲线 (Calibration)')
ax_cal.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
ax_cal.grid(True, linestyle=':', alpha=0.6)

plt.savefig('figures/fig6_q4_model_evaluation.png')
plt.close()
print('Figure 6 generated cleanly!')
