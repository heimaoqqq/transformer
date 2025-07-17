#!/usr/bin/env python3
"""
环境设置脚本 - 微多普勒时频图数据增广项目
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """运行命令并处理错误"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败:")
        print(f"错误: {e.stderr}")
        return False

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 需要 Python 3.8 或更高版本")
        return False
    print(f"✅ Python 版本: {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda():
    """检查CUDA可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用, 设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️  CUDA 不可用，将使用CPU训练")
            return False
    except ImportError:
        print("⚠️  PyTorch 未安装，无法检查CUDA")
        return False

def create_directories():
    """创建项目目录结构"""
    directories = [
        "data",
        "models", 
        "training",
        "inference",
        "utils",
        "outputs",
        "checkpoints"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 创建目录: {dir_name}")

def main():
    print("🚀 开始设置微多普勒时频图数据增广项目环境")

    # 检查Python版本
    if not check_python_version():
        return

    # 创建目录结构
    create_directories()

    # 升级pip
    run_command("python -m pip install --upgrade pip", "升级pip")

    # 安装核心依赖
    if run_command("pip install -r requirements.txt", "安装核心依赖"):
        print("✅ 核心依赖安装完成")
    else:
        print("❌ 依赖安装失败，请检查网络连接和requirements.txt文件")
        return

    # 检查CUDA
    check_cuda()

    # 运行兼容性测试
    print("\n🧪 运行兼容性测试...")
    test_result = run_command("python test_dependencies.py", "依赖兼容性测试")

    if test_result:
        print("✅ 兼容性测试通过")
    else:
        print("⚠️  兼容性测试失败，请检查错误信息")

    # 询问是否安装可选依赖
    print("\n📦 可选依赖安装")
    install_optional = input("是否安装可选依赖 (评估工具、内存优化等)? (y/n): ").strip().lower()

    if install_optional == 'y':
        run_command("python install_optional_deps.py", "可选依赖安装")

    print("\n🎉 环境设置完成！")
    print("\n📋 下一步:")
    print("1. 运行完整测试: python test_dependencies.py")
    print("2. 运行Diffusers测试: python test_diffusers_compatibility.py")
    print("3. 将您的微多普勒数据放入 data/ 目录")
    print("4. 开始训练: python train_kaggle.py --stage all")

if __name__ == "__main__":
    main()
