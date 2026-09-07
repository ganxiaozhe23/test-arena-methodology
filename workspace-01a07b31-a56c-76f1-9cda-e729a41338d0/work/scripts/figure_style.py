# -*- coding: utf-8 -*-
"""统一图表样式：中文字体 + 通用参数。"""
import os
from matplotlib import font_manager
import matplotlib.pyplot as plt

_FT = os.path.join(os.path.dirname(__file__), 'NotoSansSC-Regular.otf')
font_manager.fontManager.addfont(_FT)
plt.rcParams.update({
    'font.family': ['Noto Sans CJK SC', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'figure.dpi': 150,
    'axes.titlesize': 10,
})
