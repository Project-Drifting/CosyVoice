#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyInstaller hook for pynini library
处理 pynini、WeTextProcessing 和相关文本处理模块
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# 收集 pynini 的所有资源
try:
    datas, binaries, hiddenimports = collect_all('pynini')
except Exception:
    datas, binaries, hiddenimports = [], [], []

# 添加特定的隐藏导入
additional_hiddenimports = [
    'pynini',
    '_pywrapfst',
    'openfst_python',
    'extensions._pynini',
    
    # WeTextProcessing 相关
    'WeTextProcessing',
    'tn',
    'tn.chinese',
    'tn.chinese.normalizer',
    'tn.english',
    'tn.english.normalizer',
    'tn.processor',
    'tn.utils',
    
    # 其他文本处理相关
    'unicodedata',
    're',
    'string',
]

hiddenimports.extend(additional_hiddenimports)

# 尝试收集 WeTextProcessing 的数据文件
try:
    wtp_datas = collect_data_files('WeTextProcessing')
    datas.extend(wtp_datas)
except Exception:
    pass

# 尝试收集 tn 模块的数据文件
try:
    tn_datas = collect_data_files('tn')
    datas.extend(tn_datas)
except Exception:
    pass

# 去除重复项
hiddenimports = list(set(hiddenimports)) 