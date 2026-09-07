# test_arena — NIPT 时点选择与胎儿异常判定（数学建模求解）

本仓库基于赛题数据（男胎检测数据.csv / 女胎检测数据.csv）完成四问建模：

| 文件 | 说明 |
|---|---|
| `题目.tex` | 原始赛题 |
| `论文.md` / `论文.html` | 完整建模论文（Markdown / 自包含 HTML） |
| `论文.tex` / `论文.pdf` | LaTeX 源码与编译成品（xelatex + ctexart，25 页） |
| `figs/` | 论文全部 15 张图（fig1–fig15） |
| `solution/` | 全部可复现代码与中间结果 |
| `skills/` | 题目自带的建模求解与论文写作技能参考 |

## 模型路线（四问）

1. **问题一（相关性/关系模型）**：行级+个体级相关分析、面板固定效应、线性混合效应模型（随机截距＋随机斜率，ICC=0.81、条件 R²=0.84）——浓度随孕周 +0.0035/周、与 BMI/体重负相关、个体差异主导。
2. **问题二（BMI 分组与时点）**：BLUP+插值双路估计个体最早达标时间 τ；平滑风险函数 w(t)（11/20/29 周 = 1:4:16）下求组期望风险最小与"风险平带内达标率最高"双判据；统计检验表明 BMI=36 是唯一显著分界 → 推荐 **BMI<36：11 周首检；BMI≥36：14.5 周首检**；400–600 次参数自助给出稳健区间。
3. **问题三（多因素+达标比例+误差）**：LMM 嵌套检验表明年龄/身高/体重增量信息有限；给出达标比例 0.7–0.95 约束下的风险权衡表；量化测量噪声 σm=0.009（≈τ 的 2.6 周）与到检执行偏差的影响。
4. **问题四（女胎非整倍体判定）**：QC 门控（63.4% 通过）→ LR/RF 概率判别（GroupKFold 按孕妇分组：OOF AUC=0.803，T13 0.758 / T18 0.832 / T21 0.576）→ 等渗校准（Brier 0.158→0.074）→ 孕妇级≥2 次独立阳性复核规则（特异/PPV 100%）；并报告 AB 标签与 Z 值规则脱节、标签跨胎次不一致等数据特征及临床建议。

## 复现

```bash
cd test_arena
python3 solution/q0_eda.py        # 探索分析（figs/fig1-4）
python3 solution/q1_analysis.py   # 问题一（fig5-8, q1_results.json）
python3 solution/q2_timing.py     # 问题二（fig9-10, q2_results.json）
python3 solution/q3_multi.py      # 问题三（fig11-12, q3_results.json）
python3 solution/q4_classify.py   # 问题四-主流程（fig13-15, q4_results.json）
python3 solution/q4b_calibrate.py # 问题四-校准与复核规则
python3 solution/policy_compare.py# 表4 策略对比（政策对比.json）
python3 solution/md2html.py       # 重新生成论文.html
```

依赖：`pandas numpy scipy statsmodels scikit-learn matplotlib seaborn`。

## 主要结论速览

- 男胎 Y 浓度 4% 达标与孕周显著正相关（组内 0.0035/周），BMI≥36 孕妇中位达标时间推迟约 4 周（14.2 vs 10.0+）；
- 分组时点方案（<36 周检 11 周 / ≥36 检 14.5 周）相对经验统一 12 周方案期望风险低约 4%、复检次数更少；
- 女胎 AB 标签在现有测序特征中存在可学习的联合信号（AUC 0.80），但 T21 信号弱、单次 Z 规则与标签脱节，需"质控＋概率判定＋≥2 次独立阳性复核"的保守流程。

## LaTeX 编译

```bash
xelatex 论文.tex   # 运行两遍以更新交叉引用
```
需要 TeX Live 的 xelatex、ctex、CJK 字体（Fandol）支持。
