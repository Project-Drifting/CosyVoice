#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyInstaller hook for whisper library
处理 whisper 导入问题，避免 numba 循环导入的专用钩子
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules, is_module_satisfies

# 手动指定需要的 whisper 模块，包含所有必要的模块
hiddenimports = [
    'whisper',
    'whisper.model',
    'whisper.audio',
    'whisper.decoding',
    'whisper.tokenizer',
    'whisper.utils',
    'whisper.normalizers',
    'whisper.normalizers.basic',
    'whisper.normalizers.english',
    'whisper.timing',  # 需要包含这个模块
    'whisper.transcribe',  # 需要包含这个模块
]

# 收集数据文件但不包含有问题的模块
try:
    from PyInstaller.utils.hooks import collect_data_files
    datas = collect_data_files('whisper')
except Exception:
    datas = []

binaries = []

# 如果可能，尝试替换有问题的模块
try:
    import whisper
    # 检查是否有 timing 模块
    import whisper.timing
    # 如果导入 timing 成功但在 PyInstaller 环境中可能有问题
    # 我们可以尝试一些替代方案
except ImportError:
    # 如果 timing 模块不存在或无法导入，那很好
    pass
except Exception as e:
    # 如果有其他错误，记录但继续
    print(f"Whisper hook warning: {e}")
    pass 