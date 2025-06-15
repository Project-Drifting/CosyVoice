#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyInstaller hook for inflect library
处理 inflect 和 typeguard 相关的源代码检查问题
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

# 收集 inflect 的资源
try:
    datas, binaries, hiddenimports = collect_all('inflect')
except Exception:
    datas, binaries, hiddenimports = [], [], []

# 添加特定的隐藏导入
additional_hiddenimports = [
    'inflect',
    'inflect.engine',
    'inspect',
    'linecache',
]

# 避免包含有问题的 typeguard 装饰器
# 因为它们在 PyInstaller 环境中会尝试获取源代码而失败
problematic_modules = [
    'typeguard._decorators',
    'typeguard._functions',
]

# 合并隐藏导入，但排除有问题的模块
hiddenimports.extend(additional_hiddenimports)
hiddenimports = [imp for imp in hiddenimports if not any(prob in imp for prob in problematic_modules)]

# 去除重复项
hiddenimports = list(set(hiddenimports)) 