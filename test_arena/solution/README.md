# NIPT 时点选择与胎儿异常判定 — 数学建模完整方案

> 仓库：`https://github.com/ganxiaozhe23/test_arena`  已克隆至 `/home/user/test_arena`  
> 本方案严格参考 `skills/math-modeling-solver` 与 `skills/math-modeling-paper` 全流程（拆题→文献→模型匹配→算法展开→论文衔接）完成。

## 📁 目录结构
```
solution/
├── code/                          # 可复现代码（4问全覆盖）
│   ├── eda_and_q1.py              # Q1 EDA + OLS/R2/残差诊断
│   ├── q1_mixed_and_logistic.py   # Q1 混合线性 + Logistic + BoxCox
│   ├── q2_grouping_timing_clean.py# Q2 BMI分组 + T*插值 + 风险优化 + MC
│   ├── q3_multifactor.py          # Q3 多因素RF重要性 + 决策树 + KMeans + T*回归
│   └── q4_female_abnormal.py      # Q4 女胎19维分类（4模型对比）
├── results/
│   ├── figures/  (19张 PNG, 300dpi) # 全部图表
│   ├── q1_cleaned_male.csv
│   ├── q1_model_summaries.txt
│   └── q4_features.csv
└── paper/
    ├── NIPT时点选择与胎儿异常判定-建模论文.docx  (2.14 MB, 含19图/16表, 可打印)
    ├── NIPT时点选择与胎儿异常判定-建模论文.html  (1.4 MB, 单文件含Base64图, 支持MathJax)
    └── build_paper.py  (论文生成脚本)
```

## 🔬 四问核心结论速览

| 问题 | 数学本质 | 关键模型 | 量化结果 |
|------|---------|---------|---------|
| **Q1** Y浓度相关性 | 关联/回归+统计推断 | 混合线性 `Y=0.07+0.003GW-0.001BMI+b_i`；Box-Cox λ=0.425 | GW r=0.126 p=3e-05；BMI r=-0.151 p=6e-07；GW效应z=19.4 p<0.001；个体斜率均值0.00415/week |
| **Q2** BMI分组时点 | 聚类+优化 | K-Means轮廓0.557 + T*插值 + 风险加权 `R=w1R_time+w2(1-p)` + 蒙特卡洛 | 推荐四分组 [20,30)/[30,33)/[33,37)/[37,50) 最优t* 12.7/13.2/14.5/17.6周；高BMI组两阶段复检；MC漂移<0.3周 |
| **Q3** 多因素分组 | 聚类+优化+评价 | RF置换重要性 + KMeans多维 + T*回归 + 双阈值 | 体重95.5kg阈值比BMI更敏感；高风险簇(BMI36.5/体重99.3)达标率74.4% T*中位13.73周；提出BMI-体重双阈值策略 |
| **Q4** 女胎判定 | 分类/判别 | Logistic/RF/XGB/SVM 5折CV | 单\|Z\|>3 AUC0.479失效；融合19维后Logistic CV AUC0.808 测试0.845；阈值0.78时F1=0.516 准确率0.92；X浓度/GC最重要 |

## 📊 关键图表
- 图2 相关热力图  图3 Y-GW-BMI散点  图5 残差诊断  图6 Logistic概率  图7 个体斜率
- 图8 BMI直方图  图11-16 RF重要性/决策树  图17-21 ROC/PR/特征重要性

## 📝 论文
- **DOCX**：`paper/NIPT时点选择与胎儿异常判定-建模论文.docx` — 按国赛结构（摘要→问题分析→假设→符号→数据处理→四问建模→检验→评价→参考文献→附录），含公式、表格、流程图，可直接打印/提交。
- **HTML**：`paper/NIPT时点选择与胎儿异常判定-建模论文.html` — 单文件Base64嵌入，无需外部依赖，MathJax渲染公式，适合浏览器预览。

## ▶️ 复现
```bash
pip install statsmodels scikit-learn xgboost seaborn python-docx
python code/eda_and_q1.py
python code/q1_mixed_and_logistic.py
python code/q2_grouping_timing_clean.py
python code/q3_multifactor.py
python code/q4_female_abnormal.py
python paper/build_paper.py
```

## 📚 方法学依据
- 拆题12类本质、模型决策矩阵95+场景、Cookbook/Playbook全参考
- 假设检验（Shapiro/Levene/B-P/DW/VIF）、5折CV、蒙特卡洛敏感性、DeLong检验

## 👩‍⚕️ 临床转化建议
- **低风险** BMI<32/体重<95kg → 12–13周首检
- **中风险** 32–37/95–97kg → 13–14周
- **高风险** ≥37或≥97kg → 13周初筛 + 17–19周复检（避免单次18周的高时间风险）

---
*Generated 2026-09-07 Asia/Shanghai | Agent Mode | 完整可复现*
