#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyInstaller hook for modelscope library
处理 modelscope 动态导入问题的专用钩子
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules, is_module_satisfies

# ModelScope 的所有子模块
datas, binaries, hiddenimports = collect_all('modelscope')

# 添加特定的隐藏导入
hiddenimports += [
    'modelscope.hub.snapshot_download',
    'modelscope.hub.api',
    'modelscope.hub.constants',
    'modelscope.hub.file_download',
    'modelscope.hub.repository',
    'modelscope.hub.utils',
    'modelscope.hub.utils.utils',
    'modelscope.utils.import_utils',
    'modelscope.utils.constant',
    'modelscope.utils.config',
    'modelscope.utils.logger',
    'modelscope.utils.registry',
    'modelscope.utils.hub',
    'modelscope.utils.config_utils',
    'modelscope.utils.ast_utils',
    'modelscope.utils.file_utils',
    'modelscope.version',
    'modelscope.cli',
    'modelscope.cli.download',
]

# 收集所有 modelscope 子模块
try:
    hiddenimports += collect_submodules('modelscope')
except Exception:
    pass

# 添加与 HTTP 请求相关的依赖
hiddenimports += [
    'requests',
    'urllib3',
    'certifi',
    'charset_normalizer',
    'idna',
] 