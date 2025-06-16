#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import sys
import multiprocessing

# 在导入任何其他模块之前，过滤掉可能的 Python 参数，避免 PyInstaller 打包后的参数解析错误
def filter_python_args():
    """过滤掉所有可能的 Python 参数"""
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

# 检测是否为子进程调用
def is_subprocess():
    """检测当前是否为子进程"""
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

# 设置环境变量来减少警告信息
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # 等效于 -B
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免 tokenizers 警告
os.environ['PYTHONWARNINGS'] = 'ignore'  # 减少不必要的警告

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
    if path:
        return load_wav(str(path), sample_rate)
    return None


@app.callback(invoke_without_command=True)
def main(
    model_type: str = typer.Option(
        "cosyvoice2", "--model-type", "-m", help="选择模型类型: cosyvoice 或 cosyvoice2"
    ),
    model_dir: Path = typer.Option(
        ..., "--model-dir", "-d", exists=True, file_okay=False, dir_okay=True, help="模型目录"
    ),
    mode: str = typer.Option(
        "sft", "--mode", help="推理模式"
    ),
    text: str = typer.Option("", "--text", "-t", help="要合成的文本"),
    spk_id: str = typer.Option("", "--spk-id", help="说话人 ID，例如 '中文女'"),
    prompt_text: str = typer.Option("", "--prompt-text", help="零样本或指令模式的提示文本"),
    prompt_audio: Optional[Path] = typer.Option(
        None, "--prompt-audio", exists=True, file_okay=True, dir_okay=False, help="提示音频文件路径"
    ),
    source_audio: Optional[Path] = typer.Option(
        None, "--source-audio", exists=True, file_okay=True, dir_okay=False, help="源音频文件路径，用于语音转换"
    ),
    output: Path = typer.Option(
        Path("output.wav"), "--output", "-o", help="输出音频文件路径"
    ),
    stream: bool = typer.Option(False, "--stream", "-s", help="使用流式推理"),
    speed: float = typer.Option(1.0, "--speed", help="合成速度倍数"),
    load_jit: bool = typer.Option(False, "--load-jit", help="加载 JIT 模型"),
    load_trt: bool = typer.Option(False, "--load-trt", help="加载 TensorRT 模型"),
    load_vllm: bool = typer.Option(False, "--load-vllm", help="加载 vllm (仅 CosyVoice2)"),
    fp16: bool = typer.Option(False, "--fp16", help="使用 FP16"),
    trt_concurrent: int = typer.Option(1, "--trt-concurrent", help="TensorRT 并发数"),
    no_text_frontend: bool = typer.Option(False, "--no-text-frontend", help="不进行文本前处理"),
    print_base64: bool = typer.Option(False, "--base64", help="直接输出 base64 编码的音频内容到终端，而不保存文件")
):
    """执行 CosyVoice 合成并保存到输出文件"""
    # 确保项目根目录在 sys.path 中
    sys.path.insert(0, os.getcwd())
    # 确保第三方 Matcha-TTS 包可导入
    sys.path.insert(0, os.path.join(os.getcwd(), 'third_party', 'Matcha-TTS'))

    # 参数校验
    if model_type not in ("cosyvoice", "cosyvoice2"):
        typer.secho("错误: --model-type 必须为 cosyvoice 或 cosyvoice2", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    valid_modes = ["sft", "zero_shot", "cross_lingual", "instruct", "instruct2", "vc"]
    if mode not in valid_modes:
        typer.secho(f"错误: --mode 必须为 {', '.join(valid_modes)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 初始化模型
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

    # 加载提示和源音频为 16k，以供前端使用
    prompt_speech = load_audio(prompt_audio, 16000)
    source_speech = load_audio(source_audio, 16000)
    # 如果提示音频超过30秒，截断到前30秒以支持提取
    if prompt_speech is not None and prompt_speech.shape[1] > 30 * 16000:
        prompt_speech = prompt_speech[:, :30 * 16000]

    text_frontend = not no_text_frontend

    # 选择推理模式
    if mode == "sft":
        generator = cosy.inference_sft(text, spk_id, stream=stream, speed=speed, text_frontend=text_frontend)
    elif mode == "zero_shot":
        generator = cosy.inference_zero_shot(text, prompt_text, prompt_speech, spk_id, stream=stream, speed=speed, text_frontend=text_frontend)
    elif mode == "cross_lingual":
        generator = cosy.inference_cross_lingual(text, prompt_speech, spk_id, stream=stream, speed=speed, text_frontend=text_frontend)
    elif mode == "instruct":
        generator = cosy.inference_instruct(text, spk_id, prompt_text, stream=stream, speed=speed, text_frontend=text_frontend)
    elif mode == "instruct2":
        generator = cosy.inference_instruct2(text, prompt_text, prompt_speech, spk_id, stream=stream, speed=speed, text_frontend=text_frontend)
    elif mode == "vc":
        generator = cosy.inference_vc(source_speech, prompt_speech, stream=stream, speed=speed)
    else:
        typer.secho(f"未知模式: {mode}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 收集并合并音频片段
    segments = []
    for out in generator:
        seg = out.get("tts_speech")
        if seg is not None:
            segments.append(seg)

    if segments:
        combined = torch.cat(segments, dim=1)
        if print_base64:
            buf = io.BytesIO()
            torchaudio.save(buf, combined, cosy.sample_rate, format="wav")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode('ascii')
            typer.echo(b64)
        else:
            torchaudio.save(str(output), combined, cosy.sample_rate)
            typer.secho(f"已保存输出: {output}", fg=typer.colors.GREEN)
        
        # 成功完成后，抑制后续的警告输出并立即退出
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
        # 如果检测到是子进程且参数不完整，静默退出
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