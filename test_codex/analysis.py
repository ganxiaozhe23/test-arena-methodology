"""NIPT 数学建模分析脚本。

运行示例：
    python analysis.py --input-dir . --output-dir results --seed 20260907

脚本完成四个问题的可复现分析，并将表格、图形和 JSON 结果写入输出目录。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(20260907)

MALE_COLUMNS = [
    "sample_id", "mother_id", "age", "height", "weight", "lmp", "ivf",
    "date", "draw_count", "gest_week", "bmi", "reads", "map_rate",
    "dup_rate", "unique_reads", "gc", "z13", "z18", "z21", "zx", "zy",
    "y_conc", "x_conc", "gc13", "gc18", "gc21", "filter_rate", "aneuploidy",
    "gravidity", "parity", "healthy",
]
FEMALE_COLUMNS = [
    "sample_id", "mother_id", "age", "height", "weight", "lmp", "ivf",
    "date", "draw_count", "gest_week", "bmi", "reads", "map_rate",
    "dup_rate", "unique_reads", "gc", "z13", "z18", "z21", "zx", None,
    None, "x_conc", "gc13", "gc18", "gc21", "filter_rate", "aneuploidy",
    "gravidity", "parity", "healthy",
]

NUMERIC_COLUMNS = [
    "sample_id", "age", "height", "weight", "draw_count", "bmi", "reads",
    "map_rate", "dup_rate", "unique_reads", "gc", "z13", "z18", "z21",
    "zx", "zy", "y_conc", "x_conc", "gc13", "gc18", "gc21", "filter_rate",
    "gravidity", "parity",
]

FEATURE_LABELS = {
    "z13": "13号 Z 值", "z18": "18号 Z 值", "z21": "21号 Z 值", "zx": "X号 Z 值",
    "x_conc": "X染色体浓度", "gc": "总体 GC 含量", "gc13": "13号 GC", "gc18": "18号 GC",
    "gc21": "21号 GC", "reads": "原始读段数", "unique_reads": "唯一比对读段数",
    "map_rate": "比对比例", "dup_rate": "重复比例", "filter_rate": "过滤比例",
    "age": "年龄", "height": "身高", "weight": "体重", "bmi": "BMI", "gest_week": "孕周",
}


def parse_week(value) -> float:
    """将 11w+6、20w、16W+1 转为孕周小数。"""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    m = re.match(r"^(\d+)W(?:\+(\d+))?$", text)
    if not m:
        return np.nan
    return int(m.group(1)) + int(m.group(2) or 0) / 7.0


def canonicalize(df: pd.DataFrame, female: bool = False) -> pd.DataFrame:
    """按附件列顺序统一列名，兼容女胎的两个 Unnamed 空列。"""
    source = list(df.columns)
    names = FEMALE_COLUMNS if female else MALE_COLUMNS
    if len(source) != len(names):
        raise ValueError(f"列数异常：期望 {len(names)}，实际 {len(source)}")
    out = df.copy()
    rename = {}
    for i, name in enumerate(names):
        if name is not None:
            rename[source[i]] = name
    out = out.rename(columns=rename)
    for name in set(names):
        if name is not None and name not in out.columns:
            out[name] = np.nan
    for c in NUMERIC_COLUMNS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["gest_week"] = out["gest_week"].map(parse_week)
    out["mother_id"] = out["mother_id"].astype(str).str.strip()
    out["ivf"] = out["ivf"].fillna("未知").astype(str).str.strip()
    out["aneuploidy"] = out["aneuploidy"].fillna("").astype(str).str.strip()
    out["abnormal"] = (out["aneuploidy"] != "").astype(int)
    out["reads_log"] = np.log1p(out["reads"].clip(lower=0))
    out["quality_bad"] = (
        (out["gc"].lt(0.40) | out["gc"].gt(0.60))
        | out["map_rate"].lt(0.70)
        | out["filter_rate"].gt(0.05)
    ).astype(int)
    return out


def load_tables(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    male_raw = pd.read_csv(input_dir / "男胎检测数据.csv", encoding="utf-8")
    female_raw = pd.read_csv(input_dir / "女胎检测数据.csv", encoding="utf-8")
    return canonicalize(male_raw, False), canonicalize(female_raw, True)


def ensure_dirs(output_dir: Path) -> Tuple[Path, Path]:
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figures, tables


def configure_chinese_font() -> str:
    """注册 Windows 中文字体，避免导出的 PNG 中中文变成方框/乱码。"""
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),      # Microsoft YaHei
        Path(r"C:\Windows\Fonts\simhei.ttf"),   # SimHei
        Path(r"C:\Windows\Fonts\Deng.ttf"),     # DengXian
        Path(r"C:\Windows\Fonts\simsun.ttc"),   # SimSun
    ]
    for path in candidates:
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
                family = font_manager.FontProperties(fname=str(path)).get_name()
                plt.rcParams.update({
                    "font.family": "sans-serif",
                    "font.sans-serif": [family, "DejaVu Sans"],
                    "axes.unicode_minus": False,
                })
                return family
            except Exception:
                continue
    # Linux/无中文字体环境的降级配置；Windows 本机会优先命中上面的字体。
    plt.rcParams.update({"font.family": "sans-serif", "axes.unicode_minus": False})
    return "DejaVu Sans"


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if pd.isna(obj) if not isinstance(obj, (str, bool, list, dict, tuple)) else False:
        return None
    return obj


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(json_ready(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def audit_table(df: pd.DataFrame, name: str) -> Dict:
    numeric_summary = {}
    for c in ["gest_week", "bmi", "age", "height", "weight", "reads", "y_conc", "x_conc"]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            numeric_summary[c] = {
                "missing": int(x.isna().sum()),
                "min": float(x.min()) if x.notna().any() else None,
                "max": float(x.max()) if x.notna().any() else None,
                "mean": float(x.mean()) if x.notna().any() else None,
            }
    return {
        "name": name,
        "rows": int(len(df)),
        "mothers": int(df["mother_id"].nunique()),
        "repeat_mothers": int((df.groupby("mother_id").size() > 1).sum()),
        "max_tests_per_mother": int(df.groupby("mother_id").size().max()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v},
        "numeric_summary": numeric_summary,
        "label_distribution": df["aneuploidy"].value_counts(dropna=False).to_dict(),
    }


def robust_scale(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = np.median(np.abs(s.dropna() - med))
    scale = 1.4826 * mad if mad and np.isfinite(mad) else s.std()
    return (s - med) / (scale if scale and np.isfinite(scale) else 1.0)


def prepare_male(male: pd.DataFrame) -> pd.DataFrame:
    d = male.copy()
    d = d[d["y_conc"].notna() & d["gest_week"].notna() & d["bmi"].notna()].copy()
    d["y_conc"] = d["y_conc"].clip(1e-5, 1 - 1e-5)
    d["y_logit"] = np.log(d["y_conc"] / (1 - d["y_conc"]))
    d["attained"] = (d["y_conc"] >= 0.04).astype(int)
    for c in ["gest_week", "bmi", "age", "height", "weight", "draw_count", "gravidity", "parity", "reads_log", "map_rate", "dup_rate", "gc", "filter_rate"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        # 重复测量模型不能直接接受缺失协变量；采用全样本中位数填补，
        # 并在数据审计中保留原始缺失数量，避免静默删除同一孕妇的记录。
        if d[c].isna().any():
            d[c] = d[c].fillna(d[c].median())
    d["ivf"] = d["ivf"].fillna("未知").astype(str)
    d["ivf"] = d["ivf"].replace({"": "未知", "nan": "未知", "None": "未知"})
    d["week_c"] = robust_scale(d["gest_week"])
    d["bmi_c"] = robust_scale(d["bmi"])
    d["age_c"] = robust_scale(d["age"])
    d["height_c"] = robust_scale(d["height"])
    d["weight_c"] = robust_scale(d["weight"])
    d["draw_c"] = robust_scale(d["draw_count"])
    d["gravidity_c"] = robust_scale(d["gravidity"])
    d["parity_c"] = robust_scale(d["parity"])
    d["week_c2"] = d["week_c"] ** 2
    d["bmi_c2"] = d["bmi_c"] ** 2
    d["week_bmi"] = d["week_c"] * d["bmi_c"]
    return d


def fit_question1(male: pd.DataFrame, figures: Path, tables: Path, seed: int) -> Dict:
    d = prepare_male(male)
    formula = (
        "y_logit ~ week_c + week_c2 + bmi_c + bmi_c2 + week_bmi + age_c + "
        "height_c + weight_c + draw_c + reads_log + map_rate + dup_rate + gc + filter_rate"
    )
    ols = smf.ols(formula, data=d).fit(cov_type="HC3")
    mixed = None
    try:
        mixed = smf.mixedlm(formula, data=d, groups=d["mother_id"], re_formula="1").fit(
            reml=False, method="lbfgs", maxiter=300
        )
    except Exception:
        mixed = None
    fit = mixed if mixed is not None else ols
    pred_logit = np.asarray(fit.fittedvalues)
    pred_y = 1 / (1 + np.exp(-pred_logit))
    y = d["y_conc"].to_numpy()
    metrics = {
        "model": "mixed_effects_logit" if mixed is not None else "ols_hc3_fallback",
        "n": int(len(d)),
        "mothers": int(d["mother_id"].nunique()),
        "r2_original": float(1 - np.sum((y - pred_y) ** 2) / np.sum((y - y.mean()) ** 2)),
        "rmse_original": float(np.sqrt(np.mean((y - pred_y) ** 2))),
        "mae_original": float(np.mean(np.abs(y - pred_y))),
        "residual_sd_logit": float(np.std(np.asarray(fit.resid), ddof=1)),
    }
    coef_rows = []
    for term in fit.params.index:
        coef_rows.append(
            {
                "term": term,
                "coef": float(fit.params[term]),
                "se": float(fit.bse[term]),
                "p_value": float(fit.pvalues[term]),
                "ci_low": float(fit.conf_int().loc[term, 0]),
                "ci_high": float(fit.conf_int().loc[term, 1]),
            }
        )
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(tables / "q1_coefficients.csv", index=False, encoding="utf-8-sig")

    corr_cols = ["y_conc", "gest_week", "bmi", "age", "height", "weight", "reads_log", "gc", "filter_rate"]
    corr = d[corr_cols].corr(method="spearman")["y_conc"].drop("y_conc").sort_values(key=np.abs, ascending=False)
    corr_df = corr.rename("spearman_rho").reset_index().rename(columns={"index": "variable"})
    corr_df.to_csv(tables / "q1_spearman.csv", index=False, encoding="utf-8-sig")

    # 相关矩阵图
    plt.figure(figsize=(9, 7))
    sns.heatmap(d[corr_cols].corr(), cmap="RdBu_r", center=0, annot=True, fmt=".2f", square=True)
    plt.title("男胎 Y 浓度及主要变量 Spearman/Pearson 相关矩阵")
    plt.tight_layout()
    plt.savefig(figures / "q1_correlation_heatmap.png", dpi=300)
    plt.close()

    # 孕周-BMI 平面上的拟合等高图
    grid_w = np.linspace(d.gest_week.min(), d.gest_week.max(), 80)
    grid_b = np.linspace(d.bmi.min(), d.bmi.max(), 80)
    rows = []
    for w in grid_w:
        for b in grid_b:
            row = d.iloc[0].copy()
            row["gest_week"], row["bmi"] = w, b
            row["week_c"] = (w - d.gest_week.median()) / (1.4826 * np.median(np.abs(d.gest_week - d.gest_week.median())))
            row["bmi_c"] = (b - d.bmi.median()) / (1.4826 * np.median(np.abs(d.bmi - d.bmi.median())))
            row["week_c2"], row["bmi_c2"], row["week_bmi"] = row.week_c ** 2, row.bmi_c ** 2, row.week_c * row.bmi_c
            rows.append(row)
    grid = pd.DataFrame(rows)
    try:
        z = np.asarray(fit.predict(grid)).reshape(len(grid_w), len(grid_b))
    except Exception:
        z = np.asarray(ols.predict(grid)).reshape(len(grid_w), len(grid_b))
    z = 1 / (1 + np.exp(-z))
    W, B = np.meshgrid(grid_w, grid_b, indexing="ij")
    plt.figure(figsize=(8, 6))
    cs = plt.contourf(W, B, z, levels=16, cmap="viridis")
    plt.colorbar(cs, label="预测 Y 浓度")
    plt.scatter(d.gest_week, d.bmi, c=d.y_conc, s=8, alpha=0.18, cmap="coolwarm")
    plt.axhline(28, color="white", lw=0.8, ls="--")
    plt.xlabel("孕周（周）")
    plt.ylabel("BMI")
    plt.title("问题一：孕周与 BMI 对 Y 染色体浓度的联合影响")
    plt.tight_layout()
    plt.savefig(figures / "q1_week_bmi_surface.png", dpi=300)
    plt.close()

    # 观测值与拟合值
    plt.figure(figsize=(7, 5))
    plt.scatter(y, pred_y, s=12, alpha=0.35)
    lim = [0, max(y.max(), pred_y.max())]
    plt.plot(lim, lim, "k--", lw=1)
    plt.xlabel("观测 Y 浓度")
    plt.ylabel("拟合 Y 浓度")
    plt.title("问题一：观测值与模型拟合值")
    plt.tight_layout()
    plt.savefig(figures / "q1_observed_fitted.png", dpi=300)
    plt.close()

    # 简单 LRT：去掉孕周/BMI/交互组
    lrt = {}
    reduced_formulas = {
        "week_terms": "y_logit ~ bmi_c + bmi_c2 + age_c + height_c + weight_c + draw_c + reads_log + map_rate + dup_rate + gc + filter_rate",
        "bmi_terms": "y_logit ~ week_c + week_c2 + age_c + height_c + weight_c + draw_c + reads_log + map_rate + dup_rate + gc + filter_rate",
        "interaction": "y_logit ~ week_c + week_c2 + bmi_c + bmi_c2 + age_c + height_c + weight_c + draw_c + reads_log + map_rate + dup_rate + gc + filter_rate",
    }
    ll_full = float(fit.llf)
    for name, f in reduced_formulas.items():
        try:
            red = smf.mixedlm(f, data=d, groups=d["mother_id"], re_formula="1").fit(reml=False, method="lbfgs", maxiter=200)
            df_diff = max(1, int(len(fit.params) - len(red.params)))
            stat = max(0.0, 2 * (ll_full - float(red.llf)))
            lrt[name] = {"chi2": stat, "df": df_diff, "p_value": float(stats.chi2.sf(stat, df_diff))}
        except Exception:
            lrt[name] = {"chi2": None, "df": None, "p_value": None}

    return {
        "metrics": metrics,
        "coefficients": coef_rows,
        "spearman": corr.to_dict(),
        "likelihood_ratio_tests": lrt,
        "data_range": {"week_min": float(d.gest_week.min()), "week_max": float(d.gest_week.max()), "bmi_min": float(d.bmi.min()), "bmi_max": float(d.bmi.max())},
        "fit_object": fit,
        "prepared": d,
    }


def first_attainment(male: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mother, g in male.groupby("mother_id"):
        g = g.sort_values("gest_week")
        reached = g[g["attained"] == 1]
        first = float(reached["gest_week"].iloc[0]) if len(reached) else np.nan
        rows.append(
            {
                "mother_id": mother,
                "bmi": float(g["bmi"].median()),
                "age": float(g["age"].median()),
                "height": float(g["height"].median()),
                "weight": float(g["weight"].median()),
                "first_attainment_week": first,
                "last_week": float(g["gest_week"].max()),
                "observed_attainment_rate": float(g["attained"].mean()),
                "n_tests": int(len(g)),
            }
        )
    out = pd.DataFrame(rows)
    out["censored"] = out["first_attainment_week"].isna().astype(int)
    # 右删失样本使用观测窗口末端作为分组优化的保守代理，不把它当作已达标时间。
    out["time_for_grouping"] = out["first_attainment_week"].fillna(out["last_week"] + 1.0)
    return out


def contiguous_partition(times: pd.DataFrame, k: int, min_size: int = 20) -> Tuple[List[float], float]:
    """对按 BMI 排序的孕妇做动态规划分组，返回 BMI 切点与 SSE。"""
    d = times.sort_values("bmi").reset_index(drop=True)
    n = len(d)
    x = d["time_for_grouping"].to_numpy(float)
    prefix = np.r_[0, np.cumsum(x)]
    prefix2 = np.r_[0, np.cumsum(x * x)]

    def cost(a, b):
        # [a,b) 
        m = b - a
        s = prefix[b] - prefix[a]
        s2 = prefix2[b] - prefix2[a]
        return float(max(0.0, s2 - s * s / m))

    dp = np.full((k + 1, n + 1), np.inf)
    prev = np.full((k + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for g in range(1, k + 1):
        for j in range(g * min_size, n + 1):
            for i in range((g - 1) * min_size, j - min_size + 1):
                val = dp[g - 1, i] + cost(i, j)
                if val < dp[g, j]:
                    dp[g, j], prev[g, j] = val, i
    if not np.isfinite(dp[k, n]):
        return [], float("inf")
    cuts_idx = []
    j = n
    for g in range(k, 0, -1):
        i = prev[g, j]
        if g > 1:
            cuts_idx.append(i)
        j = i
    cuts_idx = sorted(cuts_idx)
    cuts = [float(d.iloc[i - 1]["bmi"]) for i in cuts_idx]
    return cuts, float(dp[k, n])


def choose_bmi_partition(times: pd.DataFrame) -> Dict:
    n = len(times)
    candidates = []
    for k in range(2, 6):
        cuts, sse = contiguous_partition(times, k, min_size=max(18, n // 40))
        if not cuts:
            continue
        # BIC 型惩罚，避免将每组切得过细。
        score = n * np.log(max(sse / n, 1e-9)) + (k - 1) * np.log(n) * 8
        candidates.append({"k": k, "cuts": cuts, "sse": sse, "score": score})
    chosen = min(candidates, key=lambda z: z["score"])
    bins = [-np.inf] + chosen["cuts"] + [np.inf]
    times = times.copy()
    times["group"] = pd.cut(times["bmi"], bins=bins, right=False, labels=False)
    chosen["bins"] = bins
    chosen["group_counts"] = times["group"].value_counts().sort_index().to_dict()
    chosen["group_stats"] = times.groupby("group", observed=False)["time_for_grouping"].agg(["count", "median", "mean"]).reset_index().to_dict("records")
    return chosen


def q2_gee(male: pd.DataFrame):
    formula = "attained ~ week_c + week_c2 + bmi_c + week_bmi"
    try:
        model = sm.GEE.from_formula(
            formula, groups="mother_id", data=male, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()
        )
        fit = model.fit()
        if not np.isfinite(np.asarray(fit.params, dtype=float)).all():
            raise RuntimeError("GEE 未收敛")
        return fit
    except Exception:
        return smf.glm(formula, data=male, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": male["mother_id"]})


def design_row_q2(fit, week: float, bmi: float, male: pd.DataFrame) -> np.ndarray:
    w = (week - male.gest_week.median()) / (1.4826 * np.median(np.abs(male.gest_week - male.gest_week.median())))
    b = (bmi - male.bmi.median()) / (1.4826 * np.median(np.abs(male.bmi - male.bmi.median())))
    names = list(fit.params.index)
    vals = {"Intercept": 1.0, "week_c": w, "week_c2": w * w, "bmi_c": b, "week_bmi": w * b}
    return np.array([vals.get(n, 0.0) for n in names], dtype=float)


def p_and_ci(fit, x: np.ndarray) -> Tuple[float, float, float]:
    beta = np.asarray(fit.params)
    eta = float(x @ beta)
    cov = np.asarray(fit.cov_params())
    se = float(np.sqrt(max(0.0, x @ cov @ x)))
    p = 1 / (1 + np.exp(-eta))
    lo = 1 / (1 + np.exp(-(eta - 1.96 * se)))
    hi = 1 / (1 + np.exp(-(eta + 1.96 * se)))
    return p, lo, hi


def stage_penalty(week: float) -> float:
    if week <= 12:
        return 0.15
    if week < 28:
        return 0.85
    return 2.8


def q2_recommendations(male: pd.DataFrame, fit, partition: Dict, figures: Path, tables: Path, q1: Dict, seed: int) -> Dict:
    bins = partition["bins"]
    times = first_attainment(male)
    times["group"] = pd.cut(times["bmi"], bins=bins, right=False, labels=False)
    weeks = np.arange(10.0, 25.01, 0.25)
    recs = []
    curves = []
    for group, g in times.groupby("group", observed=False):
        if len(g) == 0:
            continue
        bmid = float(g.bmi.median())
        vals = []
        for week in weeks:
            x = design_row_q2(fit, week, bmid, male)
            p, lo, hi = p_and_ci(fit, x)
            vals.append((week, p, lo, hi))
        curve = pd.DataFrame(vals, columns=["week", "p", "lo", "hi"])
        curve["group"] = int(group)
        curves.append(curve)
        reliable = curve[curve["lo"] >= 0.95]
        if len(reliable):
            best = reliable.iloc[0]
            criterion = "95%置信下界≥0.95"
        else:
            reliable = curve[curve["lo"] >= 0.90]
            best = reliable.iloc[0] if len(reliable) else curve.iloc[-1]
            criterion = "95%置信下界≥0.90" if len(reliable) else "25周窗口末端仍不足"
        lo_b = bins[int(group)]
        hi_b = bins[int(group) + 1]
        recs.append(
            {
                "group": int(group), "bmi_low": None if not np.isfinite(lo_b) else float(lo_b),
                "bmi_high": None if not np.isfinite(hi_b) else float(hi_b), "n_mothers": int(len(g)),
                "median_bmi": bmid, "recommended_week": float(best.week), "predicted_rate": float(best.p),
                "ci_low": float(best.lo), "ci_high": float(best.hi), "criterion": criterion,
                "observed_first_attainment_median": float(g["first_attainment_week"].median()),
                "observed_attainment_rate": float(g["observed_attainment_rate"].mean()),
            }
        )
    curve_df = pd.concat(curves, ignore_index=True)
    curve_df.to_csv(tables / "q2_probability_curves.csv", index=False, encoding="utf-8-sig")
    rec_df = pd.DataFrame(recs)
    rec_df.to_csv(tables / "q2_bmi_recommendations.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 5.5))
    for group, g in curve_df.groupby("group"):
        plt.plot(g.week, g.p, lw=2, label=f"组{int(group)+1}")
        plt.fill_between(g.week, g.lo, g.hi, alpha=0.12)
        r = rec_df.loc[rec_df.group == group].iloc[0]
        plt.axvline(r.recommended_week, ls="--", lw=0.8)
    plt.axhline(0.95, color="k", ls=":", lw=1, label="达标概率 0.95")
    plt.xlabel("检测孕周（周）")
    plt.ylabel("Y 浓度达到 4% 的概率")
    plt.ylim(0, 1.03)
    plt.legend(ncol=2)
    plt.title("问题二：不同 BMI 组的达标概率与推荐时点")
    plt.tight_layout()
    plt.savefig(figures / "q2_bmi_probability.png", dpi=300)
    plt.close()

    # 误差分析：用问题一残差在 logit 空间扰动观测浓度，估计推荐点附近的误判概率。
    resid_sd = float(q1["metrics"]["residual_sd_logit"])
    error_rows = []
    rng = np.random.default_rng(seed + 2)
    for _, r in rec_df.iterrows():
        x = design_row_q2(fit, r.recommended_week, r.median_bmi, male)
        p, _, _ = p_and_ci(fit, x)
        eta = math.log(p / max(1 - p, 1e-9))
        latent = rng.normal(eta, resid_sd, 3000)
        observed = 1 / (1 + np.exp(-latent))
        fp = float(np.mean((observed >= 0.04) & (1 / (1 + np.exp(-eta)) < 0.04)))
        fn = float(np.mean((observed < 0.04) & (1 / (1 + np.exp(-eta)) >= 0.04)))
        error_rows.append({"group": int(r.group), "recommended_week": float(r.recommended_week), "residual_sd_logit": resid_sd, "false_positive_proxy": fp, "false_negative_proxy": fn, "p_with_error": float(np.mean(observed >= 0.04))})
    error_df = pd.DataFrame(error_rows)
    error_df.to_csv(tables / "q2_measurement_error.csv", index=False, encoding="utf-8-sig")
    return {"partition": partition, "recommendations": recs, "error_analysis": error_rows, "gee_params": {k: float(v) for k, v in fit.params.items()}, "gee_pvalues": {k: float(v) for k, v in fit.pvalues.items()}}


def prepare_q3(male: pd.DataFrame) -> pd.DataFrame:
    d = prepare_male(male)
    return d


def q3_model(male: pd.DataFrame):
    formula = (
        "attained ~ week_c + week_c2 + bmi_c + week_bmi + age_c + height_c + weight_c + "
        "gravidity_c + parity_c + reads_log + map_rate + dup_rate + gc + filter_rate + C(ivf)"
    )
    try:
        model = sm.GEE.from_formula(formula, groups="mother_id", data=male, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
        fit = model.fit()
        if not np.isfinite(np.asarray(fit.params, dtype=float)).all():
            raise RuntimeError("GEE 未收敛")
        return fit
    except Exception:
        return smf.glm(formula, data=male, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": male["mother_id"]})


def q3_recommendations(male: pd.DataFrame, fit, q2: Dict, figures: Path, tables: Path, seed: int) -> Dict:
    d = prepare_q3(male)
    bins = q2["partition"]["bins"]
    d["group"] = pd.cut(d["bmi"], bins=bins, right=False, labels=False)
    weeks = np.arange(10.0, 25.01, 0.25)
    rng = np.random.default_rng(seed + 3)
    rows = []
    curve_rows = []
    for group, g in d.groupby("group", observed=False):
        if len(g) == 0:
            continue
        # 问题三是问题二的扩展：综合风险优化不能把检测时点提前到
        # BMI-only 模型尚未达到安全下限的位置。
        q2_group = next((x for x in q2["recommendations"] if int(x["group"]) == int(group)), None)
        safety_floor = float(q2_group["recommended_week"]) if q2_group else 12.0
        # 将每个个体的 BMI 等静态特征保留，只改变候选检测孕周。
        probs = []
        for week in weeks:
            h = g.copy()
            h["gest_week"] = week
            h["week_c"] = (week - d.gest_week.median()) / (1.4826 * np.median(np.abs(d.gest_week - d.gest_week.median())))
            h["week_c2"] = h["week_c"] ** 2
            h["week_bmi"] = h["week_c"] * h["bmi_c"]
            p = np.asarray(fit.predict(h), dtype=float)
            if not np.isfinite(p).any():
                raise RuntimeError("问题三模型预测全部为缺失，请检查模型收敛与协变量缺失值。")
            p = np.nan_to_num(p, nan=float(np.nanmedian(p)), posinf=1.0, neginf=0.0)
            # 用残差的 logistic-normal 扰动近似测量误差。
            eta = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
            noise = rng.normal(0, 0.35, size=len(p))
            p_err = float(np.mean(1 / (1 + np.exp(-(eta + noise)))))
            probs.append((week, float(np.mean(p)), p_err, float(np.std(p) / math.sqrt(max(len(p), 1)))))
        curve = pd.DataFrame(probs, columns=["week", "p", "p_error", "se"])
        curve["group"] = int(group)
        curve["risk"] = 4 * (1 - curve["p_error"]) + curve["week"].map(stage_penalty)
        curve_rows.append(curve)
        curve = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["risk"])
        if curve.empty:
            raise RuntimeError(f"问题三第 {int(group)+1} 组未得到有效风险曲线。")
        curve = curve[curve["week"] >= safety_floor].copy()
        if curve.empty:
            curve = pd.DataFrame(probs, columns=["week", "p", "p_error", "se"])
            curve["group"] = int(group)
            curve["risk"] = 4 * (1 - curve["p_error"]) + curve["week"].map(stage_penalty)
        risk_best = curve.loc[curve["risk"].idxmin()]
        reliable = curve[curve["p_error"] - 1.96 * curve["se"] >= 0.95]
        threshold_week = float(reliable.iloc[0].week) if len(reliable) else float(risk_best.week)
        lo_b = bins[int(group)]
        hi_b = bins[int(group) + 1]
        rows.append(
            {
                "group": int(group), "bmi_low": None if not np.isfinite(lo_b) else float(lo_b),
                "bmi_high": None if not np.isfinite(hi_b) else float(hi_b), "n_records": int(len(g)),
                "n_mothers": int(g.mother_id.nunique()), "recommended_week": float(risk_best.week),
                "risk_score": float(risk_best.risk), "predicted_attainment_rate": float(risk_best.p_error),
                "threshold_95_week": threshold_week, "observed_attainment_rate": float(g.attained.mean()),
                "early_fraction": float(np.mean(g.gest_week <= 12)), "mid_fraction": float(np.mean((g.gest_week > 12) & (g.gest_week < 28))),
                "late_fraction": float(np.mean(g.gest_week >= 28)),
            }
        )
    curve_df = pd.concat(curve_rows, ignore_index=True)
    rec_df = pd.DataFrame(rows)
    curve_df.to_csv(tables / "q3_risk_curves.csv", index=False, encoding="utf-8-sig")
    rec_df.to_csv(tables / "q3_integrated_recommendations.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 5.5))
    for group, g in curve_df.groupby("group"):
        plt.plot(g.week, g.risk, lw=2, label=f"组{int(group)+1}")
        r = rec_df.loc[rec_df.group == group].iloc[0]
        plt.scatter([r.recommended_week], [r.risk_score], s=45)
    plt.xlabel("检测孕周（周）")
    plt.ylabel("综合风险分数（越低越好）")
    plt.legend(ncol=2)
    plt.title("问题三：综合个体差异与检测误差的风险曲线")
    plt.tight_layout()
    plt.savefig(figures / "q3_risk_curve.png", dpi=300)
    plt.close()
    return {"recommendations": rows, "model_params": {k: float(v) for k, v in fit.params.items()}, "model_pvalues": {k: float(v) for k, v in fit.pvalues.items()}}


def female_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    feature_cols = [
        "z13", "z18", "z21", "zx", "x_conc", "gc", "gc13", "gc18", "gc21",
        "reads", "unique_reads", "map_rate", "dup_rate", "filter_rate", "age", "height", "weight", "bmi", "gest_week",
    ]
    categorical = ["ivf"]
    X = df[feature_cols + categorical].copy()
    return X, feature_cols, categorical


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict:
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else None,
        "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(tp / max(tp + fp, 1)), "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.90) -> float:
    thresholds = np.unique(np.r_[0.0, p, 1.0])
    feasible = []
    for t in thresholds:
        m = binary_metrics(y, p, float(t))
        if m["sensitivity"] >= target:
            feasible.append((m["specificity"], -t, float(t)))
    return max(feasible)[2] if feasible else 0.5


def fit_female_models(female: pd.DataFrame, figures: Path, tables: Path, seed: int) -> Dict:
    df = female.copy()
    y = df["abnormal"].to_numpy(int)
    X, numeric, categorical = female_features(df)
    groups = df["mother_id"].to_numpy()
    pre = ColumnTransformer(
        [("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
         ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)]
    )
    logit = Pipeline([("pre", pre), ("model", LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5, random_state=seed))])
    rf = Pipeline([("pre", pre), ("model", RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=4, class_weight="balanced_subsample", random_state=seed, n_jobs=-1))])
    models = {"logistic": logit, "random_forest": rf}
    oof = {name: np.full(len(df), np.nan) for name in models}
    splitter = GroupKFold(n_splits=5)
    for train, test in splitter.split(X, y, groups):
        for name, model in models.items():
            model.fit(X.iloc[train], y[train])
            oof[name][test] = model.predict_proba(X.iloc[test])[:, 1]
    # 规则基线：任一常染色体 |Z|≥3。
    rule_prob = (df[["z13", "z18", "z21"]].abs().max(axis=1) >= 3).astype(float).to_numpy()
    probs = {"z_rule": rule_prob, **oof}
    result_metrics = {}
    threshold_rows = []
    for name, p in probs.items():
        t = threshold_for_sensitivity(y, p, 0.90) if name != "z_rule" else 0.5
        result_metrics[name] = binary_metrics(y, p, t)
        result_metrics[name]["default_0.5"] = binary_metrics(y, p, 0.5)
        threshold_rows.append({"model": name, "threshold_sens90": t})
    pd.DataFrame(threshold_rows).to_csv(tables / "q4_thresholds.csv", index=False, encoding="utf-8-sig")
    metric_df = pd.DataFrame(result_metrics).T.reset_index().rename(columns={"index": "model"})
    metric_df.to_csv(tables / "q4_model_metrics.csv", index=False, encoding="utf-8-sig")

    # ROC / PR 图
    plt.figure(figsize=(7, 5.5))
    for name, p in probs.items():
        if len(np.unique(y)) > 1:
            fpr, tpr, _ = roc_curve(y, p)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y,p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率")
    plt.title("问题四：女胎异常判定 ROC 曲线（按孕妇分组交叉验证）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "q4_roc.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5.5))
    for name, p in probs.items():
        precision, recall, _ = precision_recall_curve(y, p)
        plt.plot(recall, precision, lw=2, label=f"{name} (AP={average_precision_score(y,p):.3f})")
    plt.xlabel("召回率")
    plt.ylabel("精确率")
    plt.title("问题四：女胎异常判定 PR 曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "q4_pr.png", dpi=300)
    plt.close()

    # 训练最终模型并提取系数/特征重要性。
    importance = []
    for name, model in models.items():
        model.fit(X, y)
        prep = model.named_steps["pre"]
        names = list(prep.get_feature_names_out())
        estimator = model.named_steps["model"]
        if hasattr(estimator, "coef_"):
            vals = np.abs(estimator.coef_[0])
        else:
            vals = estimator.feature_importances_
        for n, v in sorted(zip(names, vals), key=lambda z: z[1], reverse=True):
            importance.append({"model": name, "feature": n, "importance": float(v), "label": FEATURE_LABELS.get(n.split("__")[-1], n)})
    imp_df = pd.DataFrame(importance)
    imp_df.to_csv(tables / "q4_feature_importance.csv", index=False, encoding="utf-8-sig")
    top = imp_df[imp_df.model == "random_forest"].head(12).sort_values("importance")
    if len(top):
        plt.figure(figsize=(8, 5.5))
        plt.barh(top["label"], top["importance"], color="#4472C4")
        plt.xlabel("重要性")
        plt.title("问题四：随机森林特征重要性")
        plt.tight_layout()
        plt.savefig(figures / "q4_feature_importance.png", dpi=300)
        plt.close()

    # 孕妇级聚合：一位孕妇任一检测呈异常则聚合为异常。
    mother_y = df.groupby("mother_id")["abnormal"].max()
    mother_p = pd.DataFrame({name: df.assign(p=p).groupby("mother_id")["p"].max() for name, p in probs.items()})
    mother_metrics = {}
    for name in probs:
        p = mother_p[name].reindex(mother_y.index).to_numpy()
        mother_metrics[name] = binary_metrics(mother_y.to_numpy(), p, threshold_for_sensitivity(mother_y.to_numpy(), p, 0.90) if name != "z_rule" else 0.5)
    return {"sample_metrics": result_metrics, "mother_metrics": mother_metrics, "abnormal_samples": int(y.sum()), "abnormal_mothers": int(mother_y.sum()), "feature_importance": importance[:50]}


def make_summary(audit_male: Dict, audit_female: Dict, q1: Dict, q2: Dict, q3: Dict, q4: Dict) -> Dict:
    q1_summary = {k: v for k, v in q1.items() if k not in {"fit_object", "prepared"}}
    return {"audit": {"male": audit_male, "female": audit_female}, "q1": q1_summary, "q2": q2, "q3": q3, "q4": q4}


def main() -> None:
    parser = argparse.ArgumentParser(description="NIPT 四问数学建模分析")
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    global RNG
    RNG = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    sns.set_theme(style="whitegrid")
    configure_chinese_font()
    figures, tables = ensure_dirs(args.output_dir)
    male, female = load_tables(args.input_dir)
    audit_male = audit_table(male, "男胎")
    audit_female = audit_table(female, "女胎")
    dump_json(args.output_dir / "data_audit.json", {"male": audit_male, "female": audit_female})

    q1 = fit_question1(male, figures, tables, args.seed)
    male_prepared = q1["prepared"]
    q2fit = q2_gee(male_prepared)
    attainment = first_attainment(male_prepared)
    partition = choose_bmi_partition(attainment)
    q2 = q2_recommendations(male_prepared, q2fit, partition, figures, tables, q1, args.seed)
    q3fit = q3_model(male_prepared)
    q3 = q3_recommendations(male_prepared, q3fit, q2, figures, tables, args.seed)
    q4 = fit_female_models(female, figures, tables, args.seed)
    summary = make_summary(audit_male, audit_female, q1, q2, q3, q4)
    dump_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "male_rows": len(male), "female_rows": len(female), "bmi_groups": partition["k"], "q4_positive": q4["abnormal_samples"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
