# -*- coding: utf-8 -*-
"""共享的数据读取与清洗模块（NIPT 男胎 / 女胎检测数据）。"""
import os
import re
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MALE_CSV = os.path.join(ROOT, '男胎检测数据.csv')
FEMALE_CSV = os.path.join(ROOT, '女胎检测数据.csv')
WORK = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

COLS = ['id', 'mother', 'age', 'height', 'weight', 'lmp', 'ivf',
        'test_date', 'blood_draw', 'gest_raw', 'bmi',
        'reads_total', 'map_ratio', 'dup_ratio', 'uniq_reads', 'gc',
        'z13', 'z18', 'z21', 'zx', 'zy', 'conc_y', 'conc_x',
        'gc13', 'gc18', 'gc21', 'filter_ratio', 'aneuploidy',
        'n_preg', 'n_birth', 'healthy']


def parse_gest(raw):
    """'11w+6' -> 11 + 6/7 周；'13w' -> 13.0 周。"""
    m = re.match(r'(\d+)w(?:\+(\d+))?', str(raw).strip())
    if not m:
        return np.nan
    days = int(m.group(2)) if m.group(2) else 0
    return int(m.group(1)) + days / 7.0


def _base_load(path):
    df = pd.read_csv(path)
    # 统一列数（女胎 Y 相关两列为空白，列名 Unnamed）
    df = df.iloc[:, :31]
    df.columns = COLS
    return df


def load_male():
    df = _base_load(MALE_CSV)
    df['gest'] = df['gest_raw'].map(parse_gest)
    df['sex'] = 'M'
    df['test_date'] = pd.to_datetime(df['test_date'].astype(str), format='%Y%m%d', errors='coerce')
    return df


def load_female():
    df = _base_load(FEMALE_CSV)
    df['gest'] = df['gest_raw'].map(parse_gest)
    df['sex'] = 'F'
    df['zy'] = np.nan
    df['conc_y'] = np.nan
    df['test_date'] = pd.to_datetime(df['test_date'].astype(str), format='%Y%m%d', errors='coerce')
    return df


def clean_male(df):
    """男胎分析用：去掉孕周、BMI、Y 浓度缺失，去掉极端离群行。返回 (clean, log)。"""
    log = {'n_raw': len(df)}
    d = df.copy()
    d = d.dropna(subset=['gest', 'bmi', 'conc_y'])
    log['n_drop_missing'] = log['n_raw'] - len(d)
    # 离群：Y 浓度超过 0.20（远高于孕晚期典型值），或孕周超界
    out_y = (d['conc_y'] > 0.20) | (d['conc_y'] < 0.004)
    out_g = (d['gest'] < 10) | (d['gest'] > 28)
    log['n_outlier'] = int((out_y | out_g).sum())
    d = d.loc[~(out_y | out_g)]
    log['n_clean'] = len(d)
    # BMI 一致性核验（体重/身高^2）
    bmi_calc = d['weight'] / (d['height'] / 100.0) ** 2
    log['bmi_consistency_max_abs_diff'] = float((bmi_calc - d['bmi']).abs().max())
    return d.reset_index(drop=True), log


def clean_female(df):
    log = {'n_raw': len(df)}
    d = df.copy()
    d['bmi'] = d['bmi'].fillna(d['weight'] / (d['height'] / 100.0) ** 2)
    d = d.dropna(subset=['gest'])
    log['n_clean'] = len(d)
    return d.reset_index(drop=True), log


def risk_weight(t):
    """题目给定的发现时点风险权重：<=12 周低，13-27 周高，>=28 周极高。"""
    return np.where(t <= 12, 1.0, np.where(t <= 27, 3.0, 6.0))
