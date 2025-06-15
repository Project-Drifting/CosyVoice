#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyInstaller hook for numba library
处理 numba 循环导入问题的专用钩子
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules, is_module_satisfies

# 收集 numba 的所有资源，但要小心处理循环导入
try:
    datas, binaries, hiddenimports = collect_all('numba')
except Exception:
    datas, binaries, hiddenimports = [], [], []

# 明确添加需要的隐藏导入，避免循环导入
additional_hiddenimports = [
    'numba.core',
    'numba.core.types',
    'numba.core.types.common',
    'numba.core.types.scalars',
    'numba.core.types.containers',
    'numba.core.pythonapi_impl',
    'numba.core.imputils',
    'numba.core.cgutils',
    'numba.core.typing',
    'numba.core.typing.signature',
    'numba.core.typing.typeof',
    'numba.core.extending',
    'numba.core.unsafe',
    'numba.core.callconv',
    'numba.core.target_extension',
    'numba.typed',
    'numba.typed.typeddict',
    'numba.typed.typedlist',
    'numba.numpy_support',
    'numba.misc',
    'numba.misc.special',
    'numba.cpython',
    'numba.cpython.unicode',
    'numba.cpython.tupleobj',
    'numba.cpython.listobj',
    'numba.cpython.setobj',
    'numba.targets',
    'numba.targets.cpu',
    'numba.targets.registry',
    'numba.targets.base',
    'numba.targets.callconv',
    'numba.targets.cpu_target',
    'numba.targets.imputils',
    'numba.targets.mathimpl',
    'numba.targets.npyimpl',
    'numba.targets.operatorimpl',
    'numba.targets.pythonapi_impl',
    'numba.targets.rangeimpl',
    'numba.targets.setitemimpl',
    'numba.targets.slicing',
    'numba.targets.tupleimpl',
    
    # 尽量避免 experimental 相关的模块，因为它们容易引起循环导入
    # 但如果必须要，单独列出来
    'numba.experimental.function_type',
]

# 合并隐藏导入
hiddenimports.extend(additional_hiddenimports)

# 去除重复项
hiddenimports = list(set(hiddenimports))

# 特别处理一些可能导致循环导入的模块
# 我们尝试不包含这些模块，看看是否能避免问题
problematic_modules = [
    'numba.experimental.jitclass',
    'numba.experimental.jitclass.boxing',
    'numba.experimental.jitclass.base',
]

# 从 hiddenimports 中移除有问题的模块
hiddenimports = [imp for imp in hiddenimports if not any(prob in imp for prob in problematic_modules)] 