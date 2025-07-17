#!/usr/bin/env python3
"""
Kaggle快速开始脚本
在Kaggle Notebook中运行此脚本来设置和开始训练
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description=""):
    """运行命令并显示输出"""
    print(f"🔄 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print("Output:", result.stdout)
        if result.stderr:
            print("Error:", result.stderr)
            
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed")
            return False
            
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def main():
    print("🚀 Kaggle微多普勒VAE项目快速开始")
    print("=" * 50)
    
    # 检查是否在Kaggle环境
    if "/kaggle" not in os.getcwd():
        print("⚠️  Warning: 不在Kaggle环境中")
    else:
        print("✅ 检测到Kaggle环境")
    
    # 检查数据集
    data_path = Path("/kaggle/input/dataset")
    if data_path.exists():
        print(f"✅ 数据集路径存在: {data_path}")
        
        # 统计用户目录
        user_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('ID_')]
        print(f"📊 找到 {len(user_dirs)} 个用户目录")
        
        # 显示前几个用户的图像数量
        for user_dir in sorted(user_dirs, key=lambda x: int(x.name.split('_')[1]))[:5]:
            images = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg")) + list(user_dir.glob("*.jpeg"))
            print(f"   {user_dir.name}: {len(images)} 张图像")
            
    else:
        print(f"❌ 数据集路径不存在: {data_path}")
        print("请确保在Kaggle中正确添加了数据集")
        return
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU可用: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ GPU不可用")
            return
    except ImportError:
        print("⚠️  PyTorch未安装，将在后续步骤中安装")
    
    # 安装依赖
    print("\n📦 安装依赖包...")
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "diffusers>=0.25.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "pillow>=9.5.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "tqdm>=4.65.0",
        "einops>=0.7.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"安装 {package}"):
            print(f"⚠️  {package} 安装失败，继续...")
    
    # 验证环境
    print("\n🔍 验证环境...")
    run_command("python kaggle_config.py", "验证Kaggle配置")
    
    # 询问是否开始训练
    print("\n🎯 准备开始训练")
    print("预计训练时间: 6-9小时")
    print("输出目录: /kaggle/working/outputs")
    
    response = input("\n是否开始完整训练流程? (y/n): ")
    
    if response.lower() == 'y':
        print("\n🚀 开始完整训练流程...")
        
        # 运行完整训练
        success = run_command("python train_kaggle.py --stage all", "完整训练流程")
        
        if success:
            print("\n🎉 训练完成！")
            print("📁 检查输出文件:")
            run_command("ls -la /kaggle/working/outputs/", "列出输出文件")
            
            print("\n📊 生成的图像:")
            run_command("find /kaggle/working/outputs -name '*.png' | head -10", "查找生成的图像")
            
        else:
            print("\n❌ 训练失败，请检查错误信息")
    
    else:
        print("\n📋 手动训练步骤:")
        print("1. VAE训练: python train_kaggle.py --stage vae")
        print("2. 扩散训练: python train_kaggle.py --stage diffusion") 
        print("3. 生成图像: python train_kaggle.py --stage generate")
        
        print("\n📖 详细文档:")
        print("- README.md: 项目概述")
        print("- KAGGLE_README.md: Kaggle专用指南")
        print("- KAGGLE_SETUP.md: 详细设置说明")

if __name__ == "__main__":
    main()
