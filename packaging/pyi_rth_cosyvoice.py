#!/usr/bin/env python3
# PyInstaller runtime hook for CosyVoice
import sys
import os

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
third_party_path = os.path.join(current_dir, 'third_party', 'Matcha-TTS')
if third_party_path not in sys.path:
    sys.path.insert(0, third_party_path)

# 更强的 numba 导入修补 - 必须在任何其他导入之前运行
def aggressive_patch_numba():
    """积极修补 numba.experimental 以避免循环导入"""
    try:
        import types
        import importlib.util
        
        # 创建虚拟的 numba.experimental 模块
        dummy_experimental = types.ModuleType('numba.experimental')
        sys.modules['numba.experimental'] = dummy_experimental
        
        # 创建虚拟的 jitclass 子模块
        dummy_jitclass = types.ModuleType('numba.experimental.jitclass')
        sys.modules['numba.experimental.jitclass'] = dummy_jitclass
        
        # 创建虚拟的 jitclass 子模块
        dummy_jitclass_decorators = types.ModuleType('numba.experimental.jitclass.decorators')
        sys.modules['numba.experimental.jitclass.decorators'] = dummy_jitclass_decorators
        
        dummy_jitclass_boxing = types.ModuleType('numba.experimental.jitclass.boxing')
        sys.modules['numba.experimental.jitclass.boxing'] = dummy_jitclass_boxing
        
        dummy_jitclass_overloads = types.ModuleType('numba.experimental.jitclass.overloads')
        sys.modules['numba.experimental.jitclass.overloads'] = dummy_jitclass_overloads
        
        # 添加必要的虚拟函数和属性
        def dummy_jitclass(*args, **kwargs):
            """虚拟的 jitclass 装饰器"""
            def decorator(cls):
                return cls
            return decorator if args else decorator
            
        dummy_jitclass_decorators.jitclass = dummy_jitclass
        dummy_experimental.jitclass = dummy_jitclass
        
        # 添加一个虚拟的 _box 模块
        dummy_box = types.ModuleType('numba.experimental.jitclass._box')
        sys.modules['numba.experimental.jitclass._box'] = dummy_box
        
        print("Successfully patched numba.experimental modules")
        
    except Exception as e:
        print(f"Warning: Failed to patch numba aggressively: {e}")

# 修补 librosa 的 numba 依赖
def patch_librosa_numba():
    """在 librosa 导入之前修补可能的 numba 问题"""
    try:
        # 确保 numba 核心模块可用，但阻止实验性模块
        import numba
        import numba.core
        import numba.targets
        
        # 如果 librosa 尝试使用 numba.jit，确保它可用
        if not hasattr(numba, 'jit'):
            # 如果没有 jit，创建一个虚拟的
            def dummy_jit(*args, **kwargs):
                def decorator(func):
                    return func
                return decorator if args else decorator
            numba.jit = dummy_jit
            
    except Exception as e:
        print(f"Warning: Failed to patch librosa numba dependencies: {e}")

# 预处理 whisper 导入问题
def patch_whisper_timing():
    """如果可能的话，预处理 whisper.timing 模块"""
    try:
        # 先尝试正常导入
        import whisper.timing
    except ImportError:
        try:
            # 如果导入失败，创建一个最小化的替代模块
            import types
            import torch
            import numpy as np
            
            dummy_timing = types.ModuleType('whisper.timing')
            
            # 添加一些基本的函数，如果 whisper.transcribe 需要的话
            def dummy_add_word_timestamps(*args, **kwargs):
                """虚拟的 add_word_timestamps 函数"""
                return args[0] if args else None
                
            def dummy_median_filter(x, filter_width):
                """虚拟的 median_filter 函数"""
                return x
                
            dummy_timing.add_word_timestamps = dummy_add_word_timestamps
            dummy_timing.median_filter = dummy_median_filter
            
            sys.modules['whisper.timing'] = dummy_timing
            
        except Exception as e:
            print(f"Warning: Failed to create dummy whisper.timing: {e}")
    except Exception as e:
        print(f"Warning: Failed to import whisper.timing: {e}")

# 预处理 inflect/typeguard 问题
def patch_inflect_typeguard():
    """修复 inflect 和 typeguard 的源代码检查问题"""
    try:
        # 先尝试导入 typeguard
        import typeguard
        import typeguard._decorators
        
        # 替换有问题的装饰器函数
        original_typechecked = typeguard._decorators.typechecked
        
        def safe_typechecked(func):
            """安全的 typechecked 装饰器，在 PyInstaller 环境中跳过检查"""
            try:
                return original_typechecked(func)
            except (OSError, IOError):
                # 如果无法获取源代码，直接返回原函数
                return func
        
        typeguard._decorators.typechecked = safe_typechecked
        typeguard.typechecked = safe_typechecked
        
    except Exception as e:
        print(f"Warning: Failed to patch typeguard: {e}")

# 必须首先执行最激进的修补
aggressive_patch_numba()
patch_librosa_numba()
patch_whisper_timing()
patch_inflect_typeguard()

# 预导入一些关键模块
try:
    import torch
    import torch._C
except ImportError:
    pass

try:
    import modelscope.hub.snapshot_download
except ImportError:
    pass 