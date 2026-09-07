# 计算与复现说明

本目录保存 NIPT 建模题目的全部计算代码、中间结果与图表（论文位于 `../paper/`）。

## 复现步骤

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib fonttools
cd work/scripts
python3 q1_analysis.py   # 问题1：关系模型与显著性
python3 q2_analysis.py   # 问题2：BMI 分组 + 最佳时点 + 误差敏感性
python3 q3_analysis.py   # 问题3：多因素扩展
python3 q4_analysis.py   # 问题4：女胎异常判定
```

按顺序运行（q2/q3 读取 q1 的结果 JSON）。图表输出到 `figures/`（PNG+PDF），
统计结果输出到 `results/*.json`。论文编译：

```bash
cd ../paper && xelatex paper.tex && xelatex paper.tex
```

## 关键设计

- 风险权重 $w(t)$：<=12 周 1；12-27 周线性 1→3；>27 周 3→…（题目低/高/极高的连续化）。
- 测序失败 $p_{\rm fail}(t)=0.02+0.06e^{-0.5(t-10)}$，复查间隔 2 周、附加代价 0.05（均做敏感性分析）。
- 分组：一维动态规划（最小样本量 40）+ 聚类 bootstrap（200 次）报告切点不确定性。
- 女胎分类：按孕妇分组的分层交叉验证；标签为 AB 列逐次判定结果。

## 局限披露

- 女胎标签为 NIPT 自身判定（新生儿均健康），规则学习的是"检测判定"。
- T21 在当前特征下 AUC≈0.5，无可学习信号，论文中如实披露。
- 测序失败函数为假设参数，敏感性分析覆盖 0–0.2。
