#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice 打包修复脚本
解决 PyInstaller 打包后的运行时导入问题
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def test_modelscope_import():
    """测试 modelscope 导入"""
    try:
        print("测试 modelscope 导入...")
        from modelscope import snapshot_download
        print("✓ modelscope.snapshot_download 导入成功")
        return True
    except Exception as e:
        print(f"✗ modelscope 导入失败: {e}")
        return False

def test_cosyvoice_import():
    """测试 cosyvoice 导入"""
    try:
        print("测试 cosyvoice 导入...")
        sys.path.insert(0, os.path.join(os.getcwd(), 'third_party', 'Matcha-TTS'))
        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
        print("✓ CosyVoice 导入成功")
        return True
    except Exception as e:
        print(f"✗ CosyVoice 导入失败: {e}")
        return False

def create_runtime_hook():
    """创建运行时钩子"""
    hook_content = '''#!/usr/bin/env python3
# PyInstaller runtime hook for CosyVoice
import sys
import os

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
third_party_path = os.path.join(current_dir, 'third_party', 'Matcha-TTS')
if third_party_path not in sys.path:
    sys.path.insert(0, third_party_path)

# 预导入可能有问题的模块
try:
    import modelscope.hub.snapshot_download
except ImportError:
    pass

try:
    import torch._C
except ImportError:
    pass
'''
    with open('pyi_rth_cosyvoice.py', 'w') as f:
        f.write(hook_content)
    print("✓ 创建运行时钩子: pyi_rth_cosyvoice.py")

def update_spec_file():
    """更新 spec 文件"""
    spec_file = 'cosyvoice_cli.spec'
    if not os.path.exists(spec_file):
        print(f"✗ 找不到 {spec_file}")
        return False
    
    # 读取现有内容
    with open(spec_file, 'r') as f:
        content = f.read()
    
    # 添加运行时钩子
    if 'pyi_rth_cosyvoice.py' not in content:
        content = content.replace(
            "runtime_hooks=[],",
            "runtime_hooks=['pyi_rth_cosyvoice.py'],"
        )
    
    # 添加更多数据文件
    additional_datas = '''
# 添加 modelscope 配置文件
try:
    import modelscope
    modelscope_path = os.path.dirname(modelscope.__file__)
    datas.append((modelscope_path, 'modelscope'))
except ImportError:
    pass
'''
    
    if 'modelscope_path' not in content:
        content = content.replace(
            "# 添加第三方 Matcha-TTS",
            additional_datas + "\n# 添加第三方 Matcha-TTS"
        )
    
    with open(spec_file, 'w') as f:
        f.write(content)
    
    print("✓ 更新 spec 文件")
    return True

def rebuild_with_debug():
    """使用调试模式重新构建"""
    print("开始调试模式重新构建...")
    
    # 清理构建目录
    for dir_name in ['build', 'dist']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"清理目录: {dir_name}")
    
    # 使用调试模式构建
    cmd = [
        sys.executable, "-m", "PyInstaller", 
        "cosyvoice_cli.spec", 
        "--clean", 
        "--debug", "all"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ 调试模式构建成功")
        return True
    else:
        print("✗ 调试模式构建失败")
        print("错误输出:")
        print(result.stderr)
        return False

def create_wrapper_script():
    """创建包装脚本"""
    wrapper_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import traceback

def main():
    try:
        # 设置环境
        os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
        
        # 导入并运行 CLI
        from cli import app
        app()
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("尝试修复导入问题...")
        
        # 添加可能的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, 'third_party', 'Matcha-TTS'),
            current_dir,
        ]
        
        for path in possible_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # 重试导入
        try:
            from cli import app
            app()
        except Exception as retry_e:
            print(f"重试失败: {retry_e}")
            traceback.print_exc()
            sys.exit(1)
            
    except Exception as e:
        print(f"运行时错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open('cosyvoice_wrapper.py', 'w') as f:
        f.write(wrapper_content)
    
    os.chmod('cosyvoice_wrapper.py', 0o755)
    print("✓ 创建包装脚本: cosyvoice_wrapper.py")

def main():
    """主函数"""
    print("=== CosyVoice 打包修复工具 ===\n")
    
    # 检查环境
    print("1. 检查导入状态")
    modelscope_ok = test_modelscope_import()
    cosyvoice_ok = test_cosyvoice_import()
    
    if not modelscope_ok or not cosyvoice_ok:
        print("\n发现导入问题，开始修复...")
        
        # 创建修复文件
        print("\n2. 创建修复文件")
        create_runtime_hook()
        create_wrapper_script()
        
        # 更新配置
        print("\n3. 更新配置文件")
        update_spec_file()
        
        # 重新构建
        print("\n4. 重新构建")
        if rebuild_with_debug():
            print("\n✓ 修复完成!")
            print("\n使用方法:")
            print("1. 直接使用: ./dist/cosyvoice-cli/cosyvoice-cli --help")
            print("2. 使用包装脚本: python cosyvoice_wrapper.py --help")
        else:
            print("\n✗ 修复失败，请查看错误信息")
    else:
        print("\n✓ 所有导入都正常，直接重新打包")
        rebuild_with_debug()

if __name__ == "__main__":
    main() 