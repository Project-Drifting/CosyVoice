#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice 命令行工具

这是一个强大的语音合成命令行接口，支持多种推理模式：
- SFT (Speaker Finetune): 说话人微调模式
- Zero Shot: 零样本克隆模式  
- Cross Lingual: 跨语言合成模式
- Instruct: 指令控制模式
- Instruct2: 指令+零样本组合模式
- VC (Voice Conversion): 语音转换模式

支持 CosyVoice 和 CosyVoice2 两种模型类型，提供流式和非流式推理选项。
"""

import os
import sys
import multiprocessing

# 在导入任何其他模块之前，过滤掉可能的 Python 参数，避免 PyInstaller 打包后的参数解析错误
def filter_python_args():
    """
    过滤掉所有可能的 Python 解释器参数
    
    在 PyInstaller 打包后的程序中，sys.argv 可能包含 Python 解释器的参数，
    这些参数会干扰我们自己的命令行参数解析。此函数用于清理这些无关参数。
    
    Returns:
        list: 过滤后的参数列表，不包含 Python 解释器参数
    """
    # 所有可能的单字符 Python 参数
    single_char_args = {'-B', '-S', '-O', '-OO', '-s', '-E', '-t', '-u', '-v', '-W', '-X', '-q', '-I', '-b', '-c', '-d', '-i', '-m', '-P', '-R', '-Q'}
    
    # 带值的参数（这些参数后面会跟一个值）
    args_with_values = {'-c', '-m', '-W', '-X'}
    
    filtered_argv = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        # 如果是单字符Python参数，跳过
        if arg in single_char_args:
            # 如果这个参数需要值，也跳过下一个参数
            if arg in args_with_values and i + 1 < len(sys.argv):
                i += 1  # 跳过值
            i += 1
            continue
        
        # 如果是长选项形式的Python参数，跳过
        python_long_args = [
            '--version', '--help', '--debug', '--verbose', '--quiet',
            '--optimize', '--dont-write-bytecode', '--no-site', '--no-user-site',
            '--isolated', '--ignore-environment', '--unbuffered', '--hash-randomization'
        ]
        if any(arg.startswith(long_arg) for long_arg in python_long_args):
            i += 1
            continue
            
        # 保留其他参数
        filtered_argv.append(arg)
        i += 1
    
    return filtered_argv

sys.argv = filter_python_args()

def is_subprocess():
    """
    检测当前是否为子进程调用
    
    在多进程环境下，子进程可能会意外启动 CLI，此函数用于识别这种情况
    以避免子进程产生不必要的输出或错误。
    
    Returns:
        bool: True 表示当前为子进程，False 表示主进程
    """
    try:
        # 检查是否有父进程相关的环境变量
        if os.environ.get('MULTIPROCESSING_FORKED'):
            return True
        
        # 检查进程名称和参数
        if len(sys.argv) < 2:
            return True
            
        # 检查当前进程是否为多进程子进程
        try:
            current_process = multiprocessing.current_process()
            return current_process.name != 'MainProcess'
        except:
            pass
            
        return False
    except:
        return False

# 设置环境变量来减少警告信息和优化运行环境
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # 等效于 -B，不生成 .pyc 文件
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免 tokenizers 并行处理警告
os.environ['PYTHONWARNINGS'] = 'ignore'  # 减少不必要的警告输出

from pathlib import Path
from typing import Optional

import typer
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import io
import base64

app = typer.Typer(help="CosyVoice 命令行工具，支持多种推理模式和模型类型。")


def load_audio(path: Optional[Path], sample_rate: int):
    """
    加载音频文件
    
    Args:
        path: 音频文件路径，可为 None
        sample_rate: 目标采样率
        
    Returns:
        torch.Tensor or None: 加载的音频张量，如果路径为 None 则返回 None
    """
    if path:
        return load_wav(str(path), sample_rate)
    return None


@app.callback(invoke_without_command=True)
def main(
    model_type: str = typer.Option(
        "cosyvoice2", "--model-type", "-m", help="选择模型类型: cosyvoice 或 cosyvoice2"
    ),
    model_dir: Path = typer.Option(
        ..., "--model-dir", "-d", exists=True, file_okay=False, dir_okay=True, help="模型目录路径"
    ),
    mode: str = typer.Option(
        "sft", "--mode", help="""推理模式选择:

【sft - Speaker Finetune 模式】
  必需参数: --text, --spk-id
  可选参数: --speed, --stream
  说明: 使用预训练说话人进行语音合成
  支持模型: CosyVoice, CosyVoice2

【zero_shot - 零样本克隆模式】  
  必需参数: --text, --prompt-text, --prompt-audio
  可选参数: --spk-id, --speed, --stream
  说明: 基于参考音频和文本克隆说话人风格
  支持模型: CosyVoice, CosyVoice2

【cross_lingual - 跨语言模式】
  必需参数: --text, --prompt-audio
  可选参数: --spk-id, --speed, --stream  
  说明: 保持参考音频说话人特征，合成不同语言语音
  支持模型: CosyVoice-300M (非 Instruct 版本)

【instruct - 指令控制模式】
  必需参数: --text, --spk-id, --instruct-text
  可选参数: --speed, --stream
  说明: 通过文本指令控制合成的语音风格和情感
  支持模型: CosyVoice-300M-Instruct (不支持 CosyVoice2)

【instruct2 - 指令+零样本模式】
  必需参数: --text, --prompt-audio
  可选参数: --spk-id, --prompt-text, --instruct-text, --speed, --stream
  说明: 结合指令控制和零样本克隆，可使用 --prompt-text 或 --instruct-text
  支持模型: CosyVoice2

【vc - 语音转换模式】
  必需参数: --source-audio, --prompt-audio
  可选参数: --speed, --stream
  说明: 将源音频转换为参考音频的说话人风格
  注意: 此模式不需要 --text 参数
  支持模型: CosyVoice, CosyVoice2"""
    ),
    text: str = typer.Option("", "--text", "-t", help="要合成的文本内容"),
    spk_id: str = typer.Option("", "--spk-id", help="说话人 ID，用于 'sft' 和 'instruct' 模式，例如 '中文女'"),
    prompt_text: str = typer.Option("", "--prompt-text", help="参考文本，用于零样本或指令模式的语音风格控制"),
    instruct_text: str = typer.Option("", "--instruct-text", help="指令文本，用于 instruct 和 instruct2 模式控制语音风格，末尾会自动添加 '<endofprompt>'"),
    prompt_audio: Optional[Path] = typer.Option(
        None, "--prompt-audio", exists=True, file_okay=True, dir_okay=False, 
        help="参考音频文件路径 (16kHz)，用于零样本、跨语言、指令2或语音转换模式"
    ),
    source_audio: Optional[Path] = typer.Option(
        None, "--source-audio", exists=True, file_okay=True, dir_okay=False, 
        help="源音频文件路径 (16kHz)，仅用于语音转换(vc)模式"
    ),
    output: Path = typer.Option(
        Path("output.wav"), "--output", "-o", help="输出音频文件保存路径"
    ),
    stream: bool = typer.Option(False, "--stream", "-s", help="启用流式推理，实时输出音频片段"),
    speed: float = typer.Option(1.0, "--speed", help="合成速度倍数，>0，默认 1.0 为正常速度"),
    load_jit: bool = typer.Option(False, "--load-jit", help="加载 JIT 优化模型以提升推理速度"),
    load_trt: bool = typer.Option(False, "--load-trt", help="加载 TensorRT 优化模型以提升推理速度"),
    load_vllm: bool = typer.Option(False, "--load-vllm", help="加载 vLLM 优化 (仅支持 CosyVoice2)"),
    fp16: bool = typer.Option(False, "--fp16", help="使用 FP16 精度以减少显存占用和提升速度"),
    trt_concurrent: int = typer.Option(1, "--trt-concurrent", help="TensorRT 模型并发推理数量"),
    no_text_frontend: bool = typer.Option(False, "--no-text-frontend", help="跳过文本前处理，直接使用原始文本"),
    print_base64: bool = typer.Option(False, "--base64", help="将生成的音频以 base64 编码直接输出到终端，而不保存为文件")
):
    """
    CosyVoice 语音合成主函数
    
    执行语音合成并根据参数保存到文件或输出 base64 编码。
    支持多种合成模式以满足不同的应用场景需求。
    
    ==================== 各模式参数要求详解 ====================
    
    🎯 sft 模式 (Speaker Finetune)
       必需: --text, --spk-id
       示例: --mode sft --text "你好世界" --spk-id "中文女"
    
    🎯 zero_shot 模式 (零样本克隆)
       必需: --text, --prompt-text, --prompt-audio
       示例: --mode zero_shot --text "你好世界" --prompt-text "参考文本" --prompt-audio ref.wav
    
    🎯 cross_lingual 模式 (跨语言)
       必需: --text, --prompt-audio
       示例: --mode cross_lingual --text "Hello world" --prompt-audio chinese_ref.wav
    
    🎯 instruct 模式 (指令控制)
       必需: --text, --spk-id, --instruct-text
       示例: --mode instruct --text "你好世界" --spk-id "中文女" --instruct-text "请用开心的语气说话"
    
    🎯 instruct2 模式 (指令+零样本)
       必需: --text, --prompt-audio
       示例: --mode instruct2 --text "你好世界" --prompt-audio ref.wav
    
    🎯 vc 模式 (语音转换)
       必需: --source-audio, --prompt-audio
       示例: --mode vc --source-audio source.wav --prompt-audio target_style.wav
       注意: 此模式不使用 --text 参数
    
    ==================== 通用可选参数 ====================
    --stream: 启用流式推理
    --speed: 调节语速 (默认 1.0)
    --output: 指定输出文件路径
    --base64: 输出 base64 编码而非文件
    性能优化: --fp16, --load-jit, --load-trt, --load-vllm
    """
    # 确保项目根目录在 sys.path 中，以正确导入模块
    sys.path.insert(0, os.getcwd())
    # 确保第三方 Matcha-TTS 包可正确导入
    sys.path.insert(0, os.path.join(os.getcwd(), 'third_party', 'Matcha-TTS'))

    # 输入参数校验
    if model_type not in ("cosyvoice", "cosyvoice2"):
        typer.secho("错误: --model-type 必须为 cosyvoice 或 cosyvoice2", fg=typer.colors.RED)
        raise typer.Exit(code=1)
        
    valid_modes = ["sft", "zero_shot", "cross_lingual", "instruct", "instruct2", "vc"]
    if mode not in valid_modes:
        typer.secho(f"错误: --mode 必须为 {', '.join(valid_modes)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 根据模型类型初始化相应的 CosyVoice 实例
    if model_type == "cosyvoice2":
        cosy = CosyVoice2(
            str(model_dir), load_jit=load_jit, load_trt=load_trt,
            load_vllm=load_vllm, fp16=fp16, trt_concurrent=trt_concurrent
        )
    else:
        cosy = CosyVoice(
            str(model_dir), load_jit=load_jit, load_trt=load_trt,
            fp16=fp16, trt_concurrent=trt_concurrent
        )

    # 加载音频文件 - 统一使用 16kHz 采样率以保证模型兼容性
    prompt_speech = load_audio(prompt_audio, 16000)  # 参考音频，用于零样本等模式
    source_speech = load_audio(source_audio, 16000)  # 源音频，仅用于语音转换模式
    
    # 限制提示音频长度 - 如果超过30秒，截断到前30秒以支持特征提取
    if prompt_speech is not None and prompt_speech.shape[1] > 30 * 16000:
        prompt_speech = prompt_speech[:, :30 * 16000]

    # 文本前处理开关
    text_frontend = not no_text_frontend

    # 根据不同模式验证必需参数
    if mode == "sft":
        if not text or not spk_id:
            typer.secho(f"错误: sft 模式需要提供 --text 和 --spk-id 参数", fg=typer.colors.RED)
            typer.secho("示例: --mode sft --text '你好世界' --spk-id '中文女'", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
    elif mode == "zero_shot":
        if not text or not prompt_text or prompt_speech is None:
            typer.secho(f"错误: zero_shot 模式需要提供 --text, --prompt-text 和 --prompt-audio 参数", fg=typer.colors.RED)
            typer.secho("示例: --mode zero_shot --text '你好世界' --prompt-text '参考文本' --prompt-audio ref.wav", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
    elif mode == "cross_lingual":
        if not text or prompt_speech is None:
            typer.secho(f"错误: cross_lingual 模式需要提供 --text 和 --prompt-audio 参数", fg=typer.colors.RED)
            typer.secho("示例: --mode cross_lingual --text 'Hello world' --prompt-audio chinese_ref.wav", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        # 检查模型兼容性：instruct 模型不支持 cross_lingual 模式
        if hasattr(cosy, 'instruct') and cosy.instruct is True:
            typer.secho(f"错误: 跨语言模式不支持 instruct 类型的模型，请使用 CosyVoice-300M 模型", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    elif mode == "instruct":
        if not text or not spk_id or not instruct_text:
            typer.secho(f"错误: instruct 模式需要提供 --text, --spk-id 和 --instruct-text 参数", fg=typer.colors.RED)
            typer.secho("示例: --mode instruct --text '你好世界' --spk-id '中文女' --instruct-text '请用开心的语气说话'", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        # 检查模型兼容性：只有 instruct 模型支持 instruct 模式
        if hasattr(cosy, 'instruct') and cosy.instruct is False:
            typer.secho(f"错误: instruct 模式需要使用 CosyVoice-300M-Instruct 模型", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        # CosyVoice2 不支持 instruct 模式
        if model_type == "cosyvoice2":
            typer.secho(f"错误: CosyVoice2 模型不支持 instruct 模式，请使用 CosyVoice-300M-Instruct 模型", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    elif mode == "instruct2":
        if not text or prompt_speech is None or (not instruct_text and not prompt_text):
            typer.secho(f"错误: instruct2 模式需要提供 --text, --prompt-audio 和 (--instruct-text 或 --prompt-text) 参数", fg=typer.colors.RED)
            typer.secho("示例1: --mode instruct2 --text '你好世界' --instruct-text '请用温柔的语气' --prompt-audio ref.wav", fg=typer.colors.YELLOW)
            typer.secho("示例2: --mode instruct2 --text '你好世界' --prompt-text '参考文本' --prompt-audio ref.wav", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
    elif mode == "vc":
        if source_speech is None or prompt_speech is None:
            typer.secho(f"错误: vc 模式需要提供 --source-audio 和 --prompt-audio 参数", fg=typer.colors.RED)
            typer.secho("示例: --mode vc --source-audio source.wav --prompt-audio target_style.wav", fg=typer.colors.YELLOW)
            typer.secho("注意: vc 模式不需要 --text 参数", fg=typer.colors.CYAN)
            raise typer.Exit(code=1)

    # 根据不同模式选择相应的推理方法
    if mode == "sft":
        # SFT 模式：使用预训练的说话人进行合成
        generator = cosy.inference_sft(text, spk_id, stream=stream, speed=speed, text_frontend=text_frontend)
    elif mode == "zero_shot":
        # 零样本模式：基于参考音频和文本克隆说话人风格
        generator = cosy.inference_zero_shot(text, prompt_text, prompt_speech, spk_id, stream=stream, speed=speed, text_frontend=text_frontend)
    elif mode == "cross_lingual":
        # 跨语言模式：保持参考音频的说话人特征，合成不同语言的语音
        generator = cosy.inference_cross_lingual(text, prompt_speech, spk_id, stream=stream, speed=speed, text_frontend=text_frontend)
    elif mode == "instruct":
        # 指令模式：通过文本指令控制合成的语音风格和情感
        generator = cosy.inference_instruct(text, spk_id, instruct_text, stream=stream, speed=speed, text_frontend=text_frontend)
    elif mode == "instruct2":
        # 指令+零样本模式：结合指令控制和零样本克隆
        # 优先使用 instruct_text，如果没有则使用 prompt_text
        instruction_text = instruct_text if instruct_text else prompt_text
        generator = cosy.inference_instruct2(text, instruction_text, prompt_speech, spk_id, stream=stream, speed=speed, text_frontend=text_frontend)
    elif mode == "vc":
        # 语音转换模式：将源音频转换为参考音频的说话人风格
        generator = cosy.inference_vc(source_speech, prompt_speech, stream=stream, speed=speed)
    else:
        typer.secho(f"未知模式: {mode}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 收集并合并所有生成的音频片段
    segments = []
    for out in generator:
        seg = out.get("tts_speech")  # 从生成器输出中提取音频片段
        if seg is not None:
            segments.append(seg)

    # 处理生成的音频
    if segments:
        # 将所有音频片段在时间维度上连接成完整音频
        combined = torch.cat(segments, dim=1)
        
        if print_base64:
            # 输出 base64 编码到终端
            buf = io.BytesIO()
            torchaudio.save(buf, combined, cosy.sample_rate, format="wav")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode('ascii')
            typer.echo(b64)
        else:
            # 保存为音频文件
            torchaudio.save(str(output), combined, cosy.sample_rate)
            typer.secho(f"已保存输出: {output}", fg=typer.colors.GREEN)
        
        # 成功完成后，抑制后续的警告输出并立即退出
        # 这是为了避免模型清理过程中产生的无关警告信息
        try:
            import warnings
            warnings.filterwarnings("ignore")
            
            # 重定向标准错误输出到 null，避免显示清理过程的警告
            if hasattr(os, 'devnull'):
                devnull = open(os.devnull, 'w')
                sys.stderr = devnull
                
            # 立即退出程序，避免任何后续输出
            os._exit(0)
        except:
            # 如果上述方法失败，直接退出
            sys.exit(0)
            
    else:
        typer.secho("未生成任何音频片段。", fg=typer.colors.YELLOW)


if __name__ == "__main__":
    try:
        # 如果检测到是子进程且参数不完整，静默退出避免干扰
        if is_subprocess() and len(sys.argv) < 5:
            sys.exit(0)
        app()
    except typer.Exit as e:
        # 如果是子进程的参数错误，静默退出
        if is_subprocess() and e.exit_code == 2:
            sys.exit(0)
        raise
    except Exception as e:
        # 如果是子进程的其他错误，静默退出
        if is_subprocess():
            sys.exit(0)
        raise 