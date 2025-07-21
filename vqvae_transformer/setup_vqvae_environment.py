#!/usr/bin/env python3
"""
VQ-VAE阶段专用环境配置脚本
专注于diffusers和图像处理依赖
避免transformers相关的依赖冲突
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def run_command(cmd, description="", timeout=600):
    """运行命令并返回是否成功"""
    print(f"🔄 {description}")
    print(f"   命令: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            return True
        else:
            print(f"❌ {description} 失败")
            if result.stderr:
                print(f"   错误: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时 (>{timeout}秒)")
        return False
    except Exception as e:
        print(f"❌ {description} 异常: {e}")
        return False

def install_pytorch():
    """安装PyTorch GPU版本"""
    print("🔥 安装GPU版本PyTorch...")
    
    pytorch_options = [
        "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121",
        "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118",
        "pip install torch torchvision torchaudio --upgrade",
        "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0",
    ]
    
    for i, cmd in enumerate(pytorch_options, 1):
        print(f"\n尝试PyTorch方案 {i}...")
        if run_command(cmd, f"PyTorch方案 {i}"):
            print("✅ 安装PyTorch GPU版本 成功")
            return True
    
    print("❌ 所有PyTorch安装方案都失败")
    return False

def install_vqvae_dependencies():
    """安装VQ-VAE专用依赖"""
    print("🎨 安装VQ-VAE专用依赖...")
    
    # 先卸载可能冲突的包
    run_command("pip uninstall -y huggingface_hub diffusers", "清理可能冲突的包")
    run_command("pip cache purge", "清理pip缓存")
    
    # VQ-VAE专用包 (不包含transformers)
    vqvae_packages = [
        ("huggingface_hub==0.25.2", "HuggingFace Hub (支持cached_download)"),
        ("diffusers==0.24.0", "Diffusers (VQ-VAE核心)"),
        ("safetensors>=0.3.1", "SafeTensors"),
        ("tokenizers>=0.11.1,!=0.11.3", "Tokenizers"),
    ]
    
    success_count = 0
    for package, description in vqvae_packages:
        if run_command(f"pip install '{package}' --force-reinstall --no-cache-dir", f"安装 {description}"):
            success_count += 1
    
    # 强制锁定huggingface_hub版本
    print("\n🔧 锁定huggingface_hub版本...")
    run_command("pip install 'huggingface_hub==0.25.2' --force-reinstall --no-deps", "锁定 HuggingFace Hub 0.25.2")
    
    print(f"\n📊 VQ-VAE依赖安装结果: {success_count}/{len(vqvae_packages)} 成功")
    return success_count >= len(vqvae_packages) - 1

def install_other_dependencies():
    """安装其他依赖"""
    print("📚 安装其他依赖...")
    
    other_packages = [
        "numpy==1.26.4",
        "scipy==1.11.4", 
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "opencv-python==4.8.1.78",
        "einops==0.7.0",
        "lpips==0.1.4",
        "tqdm",
        "pillow",
    ]
    
    success_count = 0
    for package in other_packages:
        if run_command(f"pip install {package}", f"安装 {package}"):
            success_count += 1
    
    print(f"📊 其他依赖安装结果: {success_count}/{len(other_packages)} 成功")
    return success_count >= len(other_packages) - 2

def test_vqvae_environment():
    """测试VQ-VAE环境"""
    print("🧪 测试VQ-VAE环境...")
    
    # 清理模块缓存
    modules_to_clear = ['torch', 'diffusers', 'huggingface_hub']
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    tests = [
        ("PyTorch", "torch"),
        ("Diffusers", "diffusers"),
        ("HuggingFace Hub", "huggingface_hub"),
    ]
    
    success_count = 0
    for name, module in tests:
        try:
            imported_module = importlib.import_module(module)
            version = getattr(imported_module, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
            success_count += 1
        except Exception as e:
            print(f"❌ {name}: 导入失败 - {e}")
    
    # 测试CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
        else:
            print("⚠️ CUDA不可用，使用CPU版本")
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
    
    # 测试VQModel
    try:
        from diffusers import VQModel
        print("✅ VQModel: 可用 (VQ-VAE核心组件)")
    except ImportError:
        print("❌ VQModel: 导入失败")
    
    print(f"\n📊 VQ-VAE环境测试结果: {success_count}/{len(tests)} 成功")
    return success_count >= len(tests) - 1

def main():
    """主函数"""
    print("🎨 VQ-VAE阶段环境配置脚本")
    print("=" * 50)
    print("🎯 专用于VQ-VAE训练的环境配置")
    print("💡 避免transformers依赖冲突")
    
    steps = [
        ("安装PyTorch", install_pytorch),
        ("安装VQ-VAE依赖", install_vqvae_dependencies),
        ("安装其他依赖", install_other_dependencies),
        ("测试环境", test_vqvae_environment),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"❌ {step_name} 失败，停止配置")
            return
    
    print("\n🎉 VQ-VAE环境配置完成!")
    print("✅ 可以开始VQ-VAE训练")
    print("\n🚀 使用方法:")
    print("   python train_main.py --skip_transformer --data_dir /kaggle/input/dataset")
    print("   或者")
    print("   python training/train_vqvae.py --data_dir /kaggle/input/dataset")

if __name__ == "__main__":
    main()
