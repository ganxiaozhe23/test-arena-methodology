# NIPT 时点选择与异常判定

本目录包含题目数据、可复现分析程序和数学建模论文。原始 `题目.tex` 保持不变。

## 运行分析

```powershell
python -m pip install -r requirements.txt
python analysis.py --input-dir . --output-dir results --seed 20260907
```

结果写入：

- `results/summary.json`：四问核心结果；
- `results/tables/`：模型参数、BMI 分组、推荐时点和分类指标；
- `results/figures/`：论文使用的高分辨率图形。

## 编译论文

```powershell
xelatex 论文.tex
xelatex 论文.tex
```

生成文件为 `论文.pdf`。
