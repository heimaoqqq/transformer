#!/usr/bin/env python3
"""
Kaggle快速修复脚本 - 一键解决版本冲突
"""

import subprocess
import sys

def run_pip(command, description=""):
    """运行pip命令"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(f"pip {command}", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - 成功")
            return True
        else:
            print(f"❌ {description} - 失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - 异常: {e}")
        return False

def quick_fix():
    """快速修复版本冲突"""
    print("🚀 Kaggle快速修复工具")
    print("=" * 40)
    
    # 1. 安装PyTorch (如果需要)
    print("\n1️⃣ 确保PyTorch正确版本...")
    run_pip("install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121", 
            "安装PyTorch 2.1.0")
    
    # 2. 安装核心依赖 (指定兼容版本)
    print("\n2️⃣ 安装核心依赖...")

    # 先安装兼容的huggingface_hub
    run_pip("install 'huggingface_hub>=0.20.0,<0.25.0'", "安装兼容的 huggingface_hub")

    # 然后安装其他包
    packages = [
        "diffusers>=0.27.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "einops",
        "opencv-python"
    ]

    for package in packages:
        run_pip(f"install {package}", f"安装 {package}")
    
    # 3. 测试导入
    print("\n3️⃣ 测试导入...")
    test_modules = [
        ('torch', 'PyTorch'),
        ('diffusers', 'Diffusers'),
        ('transformers', 'Transformers')
    ]
    
    success = 0
    for module, name in test_modules:
        try:
            exec(f"import {module}")
            print(f"✅ {name} 导入成功")
            success += 1
        except Exception as e:
            print(f"❌ {name} 导入失败: {e}")
    
    # 4. 测试VAE功能
    print("\n4️⃣ 测试VAE功能...")
    try:
        from diffusers import AutoencoderKL
        import torch
        
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3, 
            latent_channels=4,
            sample_size=32
        )
        
        with torch.no_grad():
            test_input = torch.randn(1, 3, 32, 32)
            latents = vae.encode(test_input).latent_dist.sample()
            reconstructed = vae.decode(latents).sample
        
        print("✅ VAE功能测试通过")
        print(f"   输入: {test_input.shape}")
        print(f"   潜在: {latents.shape}")
        print(f"   重建: {reconstructed.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ VAE功能测试失败: {e}")
        return False

def main():
    """主函数"""
    if quick_fix():
        print("\n🎉 修复成功！")
        print("\n📋 下一步:")
        print("   python train_kaggle.py --stage all")
    else:
        print("\n⚠️  修复未完全成功，但可以尝试训练")
        print("\n📋 故障排除:")
        print("   1. 重启Kaggle内核")
        print("   2. 重新运行此脚本")
        print("   3. 或运行: python fix_versions_kaggle.py")

if __name__ == "__main__":
    main()
