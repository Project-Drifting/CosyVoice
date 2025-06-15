# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

# 项目根目录 - spec 文件在 packaging/ 目录，所以项目根目录是上一级
spec_path = os.path.abspath(SPEC)
packaging_dir = os.path.dirname(spec_path)
project_root = os.path.dirname(packaging_dir)

# 收集所有需要的数据文件和模块
datas = []
hiddenimports = []

# 添加 cosyvoice 整个包
datas.append((os.path.join(project_root, 'cosyvoice'), 'cosyvoice'))

# 添加第三方 Matcha-TTS
datas.append((os.path.join(project_root, 'third_party', 'Matcha-TTS'), 'third_party/Matcha-TTS'))

# PyTorch 相关的隐藏导入
hiddenimports.extend([
    'torch',
    'torch._C',
    'torch._C._fft',
    'torch._C._linalg',
    'torch._C._nn',
    'torch._C._sparse',
    'torch._C._special',
    'torch.nn',
    'torch.nn.functional',
    'torch.nn.modules',
    'torch.nn.modules.activation',
    'torch.nn.modules.batchnorm',
    'torch.nn.modules.conv',
    'torch.nn.modules.dropout',
    'torch.nn.modules.linear',
    'torch.nn.modules.loss',
    'torch.nn.modules.normalization',
    'torch.nn.modules.padding',
    'torch.nn.modules.pooling',
    'torch.nn.modules.rnn',
    'torch.nn.modules.transformer',
    'torch.nn.init',
    'torch.nn.utils',
    'torch.nn.utils.rnn',
    'torch.nn.utils.spectral_norm',
    'torch.nn.utils.weight_norm',
    'torch.optim',
    'torch.optim.lr_scheduler',
    'torch.utils',
    'torch.utils.data',
    'torch.utils.data.dataloader',
    'torch.utils.data.dataset',
    'torch.jit',
    'torch.jit._script',
    'torch.jit._trace',
    'torch.cuda',
    'torch.autograd',
    'torch.autograd.function',
    'torch.distributed',
    'torch.hub',
    'torchaudio',
    'torchaudio._internal',
    'torchaudio.backend',
    'torchaudio.compliance',
    'torchaudio.compliance.kaldi',
    'torchaudio.datasets',
    'torchaudio.functional',
    'torchaudio.io',
    'torchaudio.models',
    'torchaudio.pipelines',
    'torchaudio.sox_effects',
    'torchaudio.transforms',
    'torchaudio.utils',
    'torchvision',
    'torchvision.models',
    'torchvision.transforms',
])

# CosyVoice 相关的隐藏导入
hiddenimports.extend([
    'cosyvoice',
    'cosyvoice.cli',
    'cosyvoice.cli.cosyvoice',
    'cosyvoice.utils',
    'cosyvoice.utils.file_utils',
    'cosyvoice.tokenizer',
    'cosyvoice.dataset',
    'cosyvoice.hifigan',
    'cosyvoice.flow',
    'cosyvoice.llm',
    'cosyvoice.transformer',
    'cosyvoice.vllm',
])

# ModelScope 相关的隐藏导入 - 这是关键的修复
hiddenimports.extend([
    'modelscope',
    'modelscope.hub',
    'modelscope.hub.snapshot_download',
    'modelscope.hub.api',
    'modelscope.hub.constants',
    'modelscope.hub.utils',
    'modelscope.hub.utils.utils',
    'modelscope.hub.file_download',
    'modelscope.hub.repository',
    'modelscope.utils',
    'modelscope.utils.constant',
    'modelscope.utils.config',
    'modelscope.utils.logger',
    'modelscope.utils.registry',
    'modelscope.utils.import_utils',
    'modelscope.utils.hub',
    'modelscope.utils.config_utils',
    'modelscope.utils.ast_utils',
    'modelscope.utils.file_utils',
    'modelscope.utils.constant',
    'modelscope.version',
    'modelscope.cli',
    'modelscope.cli.download',
])

# Numba 相关的隐藏导入 - 只包含核心模块，避免实验性模块
hiddenimports.extend([
    'numba',
    'numba.core',
    'numba.core.types',
    'numba.core.pythonapi_impl',
    'numba.core.imputils',
    'numba.core.cgutils',
    'numba.core.typing',
    'numba.core.typing.signature',
    'numba.core.extending',
    'numba.core.unsafe',
    # 不包含 experimental 相关模块，因为它们会引起循环导入
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
])

# Whisper 相关的隐藏导入 - 小心包含需要的模块
hiddenimports.extend([
    'whisper',
    'whisper.model',
    'whisper.audio',
    'whisper.decoding',
    'whisper.tokenizer',
    'whisper.utils',
    'whisper.timing',  # 需要这个模块，但要确保 numba 问题已解决
    'whisper.transcribe',  # 需要这个模块
    'whisper.normalizers',
    'whisper.normalizers.basic',
    'whisper.normalizers.english',
])

# Pynini 和文本处理相关的隐藏导入
hiddenimports.extend([
    'pynini',
    '_pywrapfst',
    'WeTextProcessing',
    'tn',
    'tn.chinese',
    'tn.chinese.normalizer',
    'tn.english',
    'tn.english.normalizer',
    'tn.processor',
    'openfst_python',
    'extensions',
    'extensions._pynini',
])

# 其他重要的隐藏导入
hiddenimports.extend([
    'typer',
    'click',
    'rich',
    'librosa',
    'soundfile',
    'numpy',
    'scipy',
    'transformers',
    'diffusers',
    'fastapi',
    'uvicorn',
    'gradio',
    'omegaconf',
    'hydra',
    'conformer',
    'protobuf',
    'grpcio',
    'onnx',
    'onnxruntime',
    'WeTextProcessing',
    'inflect',
    'pyworld',
    'matplotlib',
    'tensorboard',
    'lightning',
    'deepspeed',
    'openai',
    'pydantic',
    # 添加 inflect 相关的依赖
    'typeguard',
    'typeguard._decorators',
    'typeguard._functions',
    'typeguard._utils',
    'inspect',
    # 添加 Matcha-TTS 相关的依赖
    'gdown',
    'wget',
    'matplotlib.pyplot',
    'pathlib',
    'warnings',
    'matcha',
    'matcha.utils',
    'matcha.utils.utils',
    'matcha.utils.pylogger',
    'matcha.models',
    'matcha.models.components',
    'matcha.models.components.flow_matching',
    'matcha.models.components.decoder',
])

# HTTP 和网络相关的隐藏导入
hiddenimports.extend([
    'requests',
    'requests.adapters',
    'requests.auth',
    'requests.cookies',
    'requests.exceptions',
    'requests.models',
    'requests.sessions',
    'requests.structures',
    'requests.utils',
    'urllib3',
    'urllib3.connection',
    'urllib3.connectionpool',
    'urllib3.exceptions',
    'urllib3.fields',
    'urllib3.filepost',
    'urllib3.poolmanager',
    'urllib3.request',
    'urllib3.response',
    'urllib3.util',
    'urllib3.util.connection',
    'urllib3.util.retry',
    'urllib3.util.ssl_',
    'urllib3.util.timeout',
    'urllib3.util.url',
    'http.client',
    'http.cookies',
    'http.server',
])

# HuggingFace 相关的隐藏导入
hiddenimports.extend([
    'huggingface_hub',
    'huggingface_hub.constants',
    'huggingface_hub.file_download',
    'huggingface_hub.hf_api',
    'huggingface_hub.repository',
    'huggingface_hub.snapshot_download',
    'huggingface_hub.utils',
])

# 添加可能的动态导入
hiddenimports.extend([
    'pkg_resources',
    'setuptools',
    'importlib_metadata',
    'yaml',
    'json',
    'pickle',
    'joblib',
    'pathlib',
    'configparser',
    'collections',
    'collections.abc',
    'functools',
    'itertools',
    'threading',
    'multiprocessing',
    'concurrent',
    'concurrent.futures',
    'asyncio',
    'ssl',
    'socket',
    'hashlib',
    'hmac',
    'base64',
    'uuid',
    'datetime',
    'time',
    'os',
    'sys',
    'io',
    'tempfile',
    'shutil',
    'zipfile',
    'tarfile',
    'gzip',
])

block_cipher = None

# 排除有问题的模块
excludes = [
    # 排除一些可能引起问题的模块
    'tkinter',
    'matplotlib.backends._backend_tk',
    'matplotlib.backends.tkagg',
    'PIL._tkinter_finder',
    # 排除一些不必要的测试模块
    'pytest',
    'test',
    'tests',
    # 排除一些开发工具
    'IPython',
    'jupyter',
    'notebook',
    # 不再排除 typeguard._decorators，让运行时钩子处理
    # 排除有问题的 numba experimental 模块
    'numba.experimental',
    'numba.experimental.jitclass',
    'numba.experimental.jitclass.base',
    'numba.experimental.jitclass.boxing',
    'numba.experimental.jitclass._box',
    # 不排除 whisper.timing，让运行时钩子处理它
]

a = Analysis(
    [os.path.join(project_root, 'cli.py')],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[packaging_dir],  # 使用 packaging 目录为钩子搜索路径
    hooksconfig={},
    runtime_hooks=[os.path.join(packaging_dir, 'pyi_rth_cosyvoice.py')],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 过滤掉一些不必要的文件来减小体积
def filter_binaries(binaries):
    # 过滤掉一些不必要的二进制文件
    filtered = []
    exclude_patterns = [
        'Qt5',  # Qt相关
        'tcl',  # Tcl相关
        'tk',   # Tk相关
    ]
    
    for binary in binaries:
        exclude = False
        for pattern in exclude_patterns:
            if pattern in binary[0].lower():
                exclude = True
                break
        if not exclude:
            filtered.append(binary)
    return filtered

a.binaries = filter_binaries(a.binaries)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='cosyvoice-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='cosyvoice-cli'
) 