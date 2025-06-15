#!/bin/bash
# 测试打包后的程序
echo "测试 CosyVoice CLI..."

# 检查可执行文件是否存在
if [ ! -f "dist/cosyvoice-cli/cosyvoice-cli" ]; then
    echo "✗ 可执行文件不存在"
    exit 1
fi

# 测试帮助信息
echo "测试帮助信息..."
./dist/cosyvoice-cli/cosyvoice-cli --help

echo "✓ 基本测试完成"

# 如果需要测试语音合成功能，可以取消下面的注释
# echo "测试语音合成功能（需要模型文件）..."
# ./dist/cosyvoice-cli/cosyvoice-cli --model-type cosyvoice2 --model-dir pretrained_models/CosyVoice2-0.5B --mode sft --text "你好，这是一个测试。" --spk-id "中文女" --output test_output.wav
