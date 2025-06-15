#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyInstaller hook for Matcha-TTS library
处理 Matcha-TTS 相关的导入和依赖问题
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# 收集 matcha 的所有资源
try:
    datas, binaries, hiddenimports = collect_all('matcha')
except Exception:
    datas, binaries, hiddenimports = [], [], []

# 添加特定的隐藏导入
additional_hiddenimports = [
    'matcha',
    'matcha.utils',
    'matcha.utils.utils',
    'matcha.utils.pylogger',
    'matcha.utils.rich_utils',
    'matcha.utils.logging_utils',
    'matcha.utils.instantiators',
    'matcha.utils.audio',
    'matcha.models',
    'matcha.models.components',
    'matcha.models.components.flow_matching',
    'matcha.models.components.decoder',
    'matcha.hifigan',
    'matcha.hifigan.models',
    'matcha.hifigan.denoiser',
    
    # 第三方依赖
    'gdown',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends',
    'matplotlib.backends._backend_agg',
    'numpy',
    'torch',
    'torchaudio',
    'soundfile',
    'librosa',
    
    # 标准库
    'pathlib',
    'warnings',
    'inspect',
    'logging',
    'io',
    'os',
    'sys',
    'typing',
    'abc',
    'omegaconf',
    'wget',
    'hydra',
    'hydra.core',
    'hydra.core.config_store',
    'rich',
    'rich.console',
    'rich.progress',
    'rich.table',
    'rich.text',
    'rich.tree',
]

hiddenimports.extend(additional_hiddenimports)

# 尝试收集 matcha 的数据文件
try:
    matcha_datas = collect_data_files('matcha')
    datas.extend(matcha_datas)
except Exception:
    pass

# 去除重复项
hiddenimports = list(set(hiddenimports)) 