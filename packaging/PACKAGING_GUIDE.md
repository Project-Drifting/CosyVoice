# CosyVoice CLI PyInstaller 打包指南

本指南将帮助你使用 PyInstaller 将 CosyVoice CLI 打包成独立的可执行文件。

## 快速开始

### 1. 自动打包（推荐）

```bash
# 运行自动打包脚本
python build_cli.py
```

这个脚本会自动：
- 检查和安装 PyInstaller
- 检查项目依赖
- 清理旧的构建文件
- 执行打包过程
- 创建测试脚本

### 2. 手动打包

如果你想手动控制打包过程：

```bash
# 1. 安装 PyInstaller
pip install pyinstaller

# 2. 使用 spec 文件打包
pyinstaller cosyvoice_cli.spec --clean

# 3. 测试打包结果
./dist/cosyvoice-cli/cosyvoice-cli --help
```

## 文件说明

- `cosyvoice_cli.spec`: PyInstaller 配置文件，包含所有依赖和配置
- `build_cli.py`: 自动化打包脚本
- `pyinstaller_hooks.py`: 处理深度学习库动态导入的钩子文件
- `test_cli.sh`: 测试脚本（自动生成）

## 打包后的使用方法

打包完成后，可执行文件位于 `dist/cosyvoice-cli/` 目录下：

```bash
# 查看帮助
./dist/cosyvoice-cli/cosyvoice-cli --help

# SFT 模式示例
./dist/cosyvoice-cli/cosyvoice-cli \
    --model-dir /path/to/model \
    --mode sft \
    --text "你好，世界" \
    --spk-id "中文女" \
    --output output.wav

# 零样本模式示例
./dist/cosyvoice-cli/cosyvoice-cli \
    --model-dir /path/to/model \
    --mode zero_shot \
    --text "Hello, world" \
    --prompt-text "This is a demo" \
    --prompt-audio prompt.wav \
    --output output.wav

# 指令模式示例
./dist/cosyvoice-cli/cosyvoice-cli \
    --model-dir /path/to/model \
    --mode instruct \
    --text "请说这句话" \
    --prompt-text "请用温柔的语调" \
    --spk-id "中文女" \
    --output output.wav
```

## 常见问题和解决方案

### 1. 导入错误 (ImportError)

**问题**: 运行时出现模块找不到的错误

**解决方案**:
- 检查 `cosyvoice_cli.spec` 中的 `hiddenimports` 列表
- 添加缺失的模块到隐藏导入列表
- 重新打包

### 2. 动态库加载失败

**问题**: PyTorch 或其他 C++ 扩展加载失败

**解决方案**:
```bash
# 在 macOS 上可能需要设置库路径
export DYLD_LIBRARY_PATH=./dist/cosyvoice-cli:$DYLD_LIBRARY_PATH

# 在 Linux 上可能需要设置库路径
export LD_LIBRARY_PATH=./dist/cosyvoice-cli:$LD_LIBRARY_PATH
```

### 3. 模型文件路径问题

**问题**: 找不到模型文件或配置文件

**解决方案**:
- 使用绝对路径指定模型目录
- 确保模型文件权限正确
- 检查模型目录结构是否完整

### 4. 内存不足

**问题**: 打包过程中内存不足

**解决方案**:
- 使用 `--onedir` 模式而不是 `--onefile`
- 增加系统虚拟内存
- 在更高配置的机器上进行打包

### 5. 权限问题

**问题**: 可执行文件没有执行权限

**解决方案**:
```bash
chmod +x ./dist/cosyvoice-cli/cosyvoice-cli
```

## 高级配置

### 1. 自定义隐藏导入

如果遇到新的导入错误，可以在 `cosyvoice_cli.spec` 中添加：

```python
hiddenimports.extend([
    'your.missing.module',
    'another.module',
])
```

### 2. 排除不必要的模块

为了减小打包体积，可以排除一些模块：

```python
excludes = [
    'tkinter',
    'matplotlib.backends._backend_tk',
    'PIL._tkinter_finder',
]
```

### 3. 添加数据文件

如果需要包含额外的数据文件：

```python
datas.append(('/path/to/data/file', 'destination/path'))
```

## 性能优化

### 1. 减小打包体积

```bash
# 使用 UPX 压缩（如果安装了 UPX）
pyinstaller cosyvoice_cli.spec --clean --upx-dir=/path/to/upx
```

### 2. 加快启动速度

- 避免在启动时导入大型库
- 使用延迟导入
- 预编译常用模块

## 分发注意事项

### 1. 系统兼容性

- 在目标操作系统上进行打包
- 测试不同版本的操作系统
- 注意不同架构（x86, ARM）的兼容性

### 2. 依赖管理

- 确保目标机器有必要的系统库
- 考虑静态链接重要的依赖
- 提供安装脚本处理系统依赖

### 3. 许可证

- 检查所有依赖库的许可证
- 确保打包分发符合许可证要求
- 包含必要的许可证文件

## 故障排除

### 1. 启用调试模式

```bash
# 启用调试输出
./dist/cosyvoice-cli/cosyvoice-cli --help --debug

# 或者在打包时启用调试
pyinstaller cosyvoice_cli.spec --clean --debug
```

### 2. 查看依赖关系

```bash
# 分析导入依赖
pyi-archive_viewer dist/cosyvoice-cli/cosyvoice-cli

# 查看打包日志
cat build/cosyvoice-cli/warn-cosyvoice-cli.txt
```

### 3. 逐步测试

```bash
# 测试基本功能
python -c "from cosyvoice.cli.cosyvoice import CosyVoice; print('导入成功')"

# 测试模型加载
python -c "from cosyvoice.cli.cosyvoice import CosyVoice; CosyVoice('/path/to/model')"
```

## 联系支持

如果遇到无法解决的问题：

1. 检查 PyInstaller 官方文档
2. 查看 CosyVoice 项目 Issues
3. 搜索相关错误信息
4. 提供详细的错误日志和环境信息

---

**注意**: 
- 首次运行可能需要较长时间，特别是模型加载
- 确保有足够的磁盘空间（至少 5GB）
- 建议在干净的 Python 环境中进行打包 