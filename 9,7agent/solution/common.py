# -*- coding: utf-8 -*-
"""NIPT 赛题通用预处理与常量定义
将男胎/女胎原始 CSV 解析为统一格式的干净数据框:
- 孕周 解析为小数周 (如 11w+6 -> 11.8571)
- 达标(Y>=4%)指示变量
- 列名统一为英文缩写，便于建模
"""
import re
import numpy as np
import pandas as pd

DATA_DIR = "."
MALE_CSV = f"{DATA_DIR}/男胎检测数据.csv"
FEMALE_CSV = f"{DATA_DIR}/女胎检测数据.csv"

Y_TH = 0.04  # Y 染色体浓度达标阈值 4%

def parse_week(s):
    """'11w+6'/'13w'/'10w+3'/'16W+1' -> 周数(小数); 无法解析返回 NaN"""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    m = re.fullmatch(r"(\d+)\s*w(?:\+\s*(\d+))?", s, flags=re.IGNORECASE)
    if not m:
        return np.nan
    w = int(m.group(1))
    d = int(m.group(2)) if m.group(2) else 0
    return w + d / 7.0

def load_male(path=MALE_CSV):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    ren = {
        "序号": "sid", "孕妇代码": "pid", "年龄": "age", "身高": "height", "体重": "weight",
        "末次月经": "lmp", "IVF妊娠": "ivf", "检测日期": "date", "检测抽血次数": "draw",
        "检测孕周": "week_str", "孕妇BMI": "bmi", "原始读段数": "raw_reads",
        "在参考基因组上比对的比例": "map_ratio", "重复读段的比例": "dup_ratio",
        "唯一比对的读段数": "uniq_reads", "GC含量": "gc", "13号染色体的Z值": "z13",
        "18号染色体的Z值": "z18", "21号染色体的Z值": "z21", "X染色体的Z值": "zx",
        "Y染色体的Z值": "zy", "Y染色体浓度": "y_conc", "X染色体浓度": "x_conc",
        "13号染色体的GC含量": "gc13", "18号染色体的GC含量": "gc18",
        "21号染色体的GC含量": "gc21", "被过滤掉读段数的比例": "flt_ratio",
        "染色体的非整倍体": "aneu", "怀孕次数": "gravida", "生产次数": "para",
        "胎儿是否健康": "healthy",
    }
    df = df.rename(columns=ren)
    df["week"] = df["week_str"].map(parse_week)
    df["y_pass"] = (df["y_conc"] >= Y_TH).astype(int)
    # 身高体重重新校验 BMI (题给 BMI 为准, 但可对照)
    return df

def load_female(path=FEMALE_CSV):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    ren = {
        "序号": "sid", "孕妇代码": "pid", "年龄": "age", "身高": "height", "体重": "weight",
        "末次月经": "lmp", "IVF妊娠": "ivf", "检测日期": "date", "检测抽血次数": "draw",
        "检测孕周": "week_str", "孕妇BMI": "bmi", "原始读段数": "raw_reads",
        "在参考基因组上比对的比例": "map_ratio", "重复读段的比例": "dup_ratio",
        "唯一比对的读段数": "uniq_reads", "GC含量": "gc", "13号染色体的Z值": "z13",
        "18号染色体的Z值": "z18", "21号染色体的Z值": "z21", "X染色体的Z值": "zx",
        "X染色体浓度": "x_conc", "13号染色体的GC含量": "gc13",
        "18号染色体的GC含量": "gc18", "21号染色体的GC含量": "gc21",
        "被过滤掉读段数的比例": "flt_ratio", "染色体的非整倍体": "aneu",
        "怀孕次数": "gravida", "生产次数": "para", "胎儿是否健康": "healthy",
    }
    keep = ["sid", "pid", "age", "height", "weight", "lmp", "ivf", "date", "draw",
            "week_str", "bmi", "raw_reads", "map_ratio", "dup_ratio", "uniq_reads",
            "gc", "z13", "z18", "z21", "zx", "x_conc", "gc13", "gc18", "gc21",
            "flt_ratio", "aneu", "gravida", "para", "healthy"]
    df = df.rename(columns=ren)
    for c in df.columns:
        if c not in ren.values():
            df = df.drop(columns=[c])
    df["week"] = df["week_str"].map(parse_week)
    return df

def risk_stage(week):
    """按题目给出的三期风险界定发现孕周的风险级别(1低/2高/3极高)"""
    if week <= 12:
        return 1
    elif week <= 27:
        return 2
    else:
        return 3

def risk_level_value(week, base=1.0, high=4.0, extreme=16.0):
    """风险等级 -> 风险权值(低:高:极高 = 1:4:16, 指数放大体现治疗窗口损失的非线性)"""
    if week <= 12:
        return base
    elif week <= 27:
        return high
    else:
        return extreme
