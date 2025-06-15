#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyInstaller 钩子文件
处理 PyTorch、transformers 等深度学习库的动态导入问题
"""

import os
import sys
from pathlib import Path

def get_torch_hidden_imports():
    """获取 PyTorch 相关的隐藏导入"""
    imports = [
        # PyTorch 核心
        'torch._C',
        'torch._C._fft',
        'torch._C._linalg',
        'torch._C._nested',
        'torch._C._nn',
        'torch._C._sparse',
        'torch._C._special',
        'torch._dynamo',
        'torch.distributed',
        'torch.distributed.algorithms',
        'torch.distributed.elastic',
        'torch.distributed.launcher',
        'torch.distributed.nn',
        'torch.distributed.optim',
        'torch.distributed.pipeline',
        'torch.distributed.rpc',
        'torch.distributed.utils',
        'torch.fx',
        'torch.jit._script',
        'torch.jit._trace',
        'torch.jit._state',
        'torch.library',
        'torch.masked',
        'torch.multiprocessing',
        'torch.nn.parallel',
        'torch.onnx',
        'torch.overrides',
        'torch.profiler',
        'torch.quantization',
        'torch.sparse',
        'torch.utils.benchmark',
        'torch.utils.bottleneck',
        'torch.utils.checkpoint',
        'torch.utils.cpp_extension',
        'torch.utils.data.datapipes',
        'torch.utils.tensorboard',
        'torch.xpu',
        
        # TorchAudio
        'torchaudio._extension',
        'torchaudio.backend',
        'torchaudio.io',
        'torchaudio.kaldi_io',
        'torchaudio.sox_effects',
        'torchaudio.utils',
        
        # TorchVision (如果使用)
        'torchvision.ops',
        'torchvision.models',
        'torchvision.datasets',
    ]
    return imports

def get_transformers_hidden_imports():
    """获取 transformers 相关的隐藏导入"""
    imports = [
        'transformers.models',
        'transformers.models.auto',
        'transformers.models.bert',
        'transformers.models.gpt2',
        'transformers.models.wav2vec2',
        'transformers.models.hubert',
        'transformers.models.whisper',
        'transformers.pipelines',
        'transformers.tokenization_utils',
        'transformers.tokenization_utils_base',
        'transformers.trainer',
        'transformers.training_args',
        'transformers.integrations',
        'transformers.utils',
        'transformers.generation',
        'transformers.deepspeed',
        'transformers.optimization',
    ]
    return imports

def get_audio_processing_imports():
    """获取音频处理相关的隐藏导入"""
    imports = [
        # Librosa
        'librosa.core',
        'librosa.feature',
        'librosa.effects',
        'librosa.filters',
        'librosa.onset',
        'librosa.beat',
        'librosa.tempo',
        'librosa.decompose',
        'librosa.segment',
        'librosa.util',
        
        # SoundFile
        'soundfile',
        '_soundfile_data',
        
        # PyWorld
        'pyworld',
        
        # 其他音频库
        'resampy',
        'audioread',
        'scipy.io.wavfile',
        'scipy.signal',
        'scipy.fft',
        'scipy.sparse',
        'scipy.linalg',
    ]
    return imports

def get_ml_framework_imports():
    """获取机器学习框架相关的隐藏导入"""
    imports = [
        # ONNX
        'onnx',
        'onnx.helper',
        'onnx.mapping',
        'onnx.numpy_helper',
        'onnx.shape_inference',
        'onnx.version_converter',
        
        # ONNX Runtime
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.capi.onnxruntime_pybind11_state',
        
        # TensorRT (如果在 Linux 上)
        'tensorrt',
        
        # OpenVINO (如果使用)
        'openvino',
        
        # NumPy
        'numpy.core',
        'numpy.core._multiarray_umath',
        'numpy.core._multiarray_tests',
        'numpy.linalg._umath_linalg',
        'numpy.fft',
        'numpy.polynomial',
        'numpy.random',
        'numpy.distutils',
    ]
    return imports

def get_web_framework_imports():
    """获取 Web 框架相关的隐藏导入"""
    imports = [
        # FastAPI
        'fastapi.routing',
        'fastapi.middleware',
        'fastapi.security',
        'fastapi.responses',
        'fastapi.encoders',
        'fastapi.exceptions',
        'fastapi.param_functions',
        
        # Uvicorn
        'uvicorn.main',
        'uvicorn.config',
        'uvicorn.server',
        'uvicorn.protocols',
        
        # Gradio
        'gradio.interface',
        'gradio.components',
        'gradio.blocks',
        'gradio.routes',
        'gradio.utils',
        
        # Pydantic
        'pydantic.main',
        'pydantic.fields',
        'pydantic.validators',
        'pydantic.typing',
        'pydantic.utils',
    ]
    return imports

def get_text_processing_imports():
    """获取文本处理相关的隐藏导入"""
    imports = [
        # WeTextProcessing
        'WeTextProcessing',
        
        # Inflect
        'inflect',
        
        # Protobuf
        'google.protobuf',
        'google.protobuf.internal',
        'google.protobuf.pyext',
        
        # YAML
        'yaml',
        'yaml.constructor',
        'yaml.loader',
        'yaml.representer',
        'yaml.resolver',
        
        # OmegaConf
        'omegaconf',
        'omegaconf.listconfig',
        'omegaconf.dictconfig',
        
        # Hydra
        'hydra',
        'hydra.core',
        'hydra.utils',
        'hydra._internal',
    ]
    return imports

def get_all_hidden_imports():
    """获取所有隐藏导入"""
    all_imports = []
    all_imports.extend(get_torch_hidden_imports())
    all_imports.extend(get_transformers_hidden_imports())
    all_imports.extend(get_audio_processing_imports())
    all_imports.extend(get_ml_framework_imports())
    all_imports.extend(get_web_framework_imports())
    all_imports.extend(get_text_processing_imports())
    
    # 去重
    return list(set(all_imports))

def get_torch_data_files():
    """获取 PyTorch 需要包含的数据文件"""
    try:
        import torch
        torch_dir = Path(torch.__file__).parent
        
        data_files = []
        
        # 包含 PyTorch 的动态库
        lib_patterns = ['*.so', '*.dll', '*.pyd', '*.dylib']
        for pattern in lib_patterns:
            for lib_file in torch_dir.glob(f'**/{pattern}'):
                if lib_file.is_file():
                    relative_path = lib_file.relative_to(torch_dir)
                    data_files.append((str(lib_file), f'torch/{relative_path}'))
        
        return data_files
        
    except ImportError:
        return []

if __name__ == "__main__":
    print("PyInstaller 钩子配置")
    print("隐藏导入数量:", len(get_all_hidden_imports()))
    print("前10个隐藏导入:")
    for imp in get_all_hidden_imports()[:10]:
        print(f"  - {imp}") 