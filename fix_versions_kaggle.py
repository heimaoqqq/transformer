#!/usr/bin/env python3
"""
Kaggle版本修复脚本 - 查找兼容版本并安装
"""

import subprocess
import sys

def run_command(cmd, description=""):
    """运行命令并显示结果"""
    print(f"🔄 {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} - 成功")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} - 失败")
            if result.stderr:
                print(f"错误: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ {description} - 异常: {e}")
        return False

def check_available_versions():
    """检查可用版本"""
    print("🔍 检查可用版本...")
    
    packages = ['diffusers', 'transformers', 'accelerate', 'huggingface_hub']
    
    for package in packages:
        print(f"\n📦 检查 {package} 可用版本:")
        cmd = f"pip index versions {package}"
        run_command(cmd, f"查询 {package} 版本")

def install_compatible_versions():
    """安装兼容版本"""
    print("\n🔧 安装兼容版本...")
    
    # 第一步：确保PyTorch正确
    print("\n1️⃣ 确认PyTorch版本...")
    run_command("pip show torch", "检查当前PyTorch")
    
    # 第二步：安装diffusers (使用更新的版本)
    print("\n2️⃣ 安装Diffusers...")
    diffusers_versions = [
        "0.30.0",  # 较新版本
        "0.29.0", 
        "0.28.0",
        "0.27.0",
        "0.26.0"
    ]
    
    for version in diffusers_versions:
        if run_command(f"pip install diffusers=={version}", f"安装 diffusers {version}"):
            print(f"✅ Diffusers {version} 安装成功")
            break
    else:
        # 如果特定版本都失败，尝试最新版本
        run_command("pip install diffusers", "安装最新版 diffusers")
    
    # 第三步：安装transformers
    print("\n3️⃣ 安装Transformers...")
    transformers_versions = [
        "4.36.2",
        "4.35.0", 
        "4.34.0",
        "4.33.0"
    ]
    
    for version in transformers_versions:
        if run_command(f"pip install transformers=={version}", f"安装 transformers {version}"):
            print(f"✅ Transformers {version} 安装成功")
            break
    else:
        run_command("pip install transformers", "安装最新版 transformers")
    
    # 第四步：安装accelerate
    print("\n4️⃣ 安装Accelerate...")
    accelerate_versions = [
        "0.25.0",
        "0.24.0",
        "0.23.0",
        "0.22.0"
    ]
    
    for version in accelerate_versions:
        if run_command(f"pip install accelerate=={version}", f"安装 accelerate {version}"):
            print(f"✅ Accelerate {version} 安装成功")
            break
    else:
        run_command("pip install accelerate", "安装最新版 accelerate")
    
    # 第五步：安装huggingface_hub
    print("\n5️⃣ 安装HuggingFace Hub...")
    hub_versions = [
        "0.19.4",
        "0.20.0",
        "0.21.0",
        "0.22.0"
    ]
    
    for version in hub_versions:
        if run_command(f"pip install huggingface_hub=={version}", f"安装 huggingface_hub {version}"):
            print(f"✅ HuggingFace Hub {version} 安装成功")
            break
    else:
        run_command("pip install huggingface_hub", "安装最新版 huggingface_hub")

def test_imports():
    """测试导入"""
    print("\n🧪 测试导入...")
    
    # 清理模块缓存
    modules_to_clear = [
        'torch', 'torchvision', 'diffusers', 'transformers', 
        'accelerate', 'huggingface_hub'
    ]
    
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
    
    # 测试导入
    test_cases = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('diffusers', 'Diffusers'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
        ('huggingface_hub', 'HuggingFace Hub')
    ]
    
    success_count = 0
    
    for module_name, display_name in test_cases:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {display_name}: 导入失败 - {e}")
        except Exception as e:
            print(f"⚠️  {display_name}: 导入异常 - {e}")
    
    print(f"\n📊 导入测试: {success_count}/{len(test_cases)} 成功")
    return success_count == len(test_cases)

def test_diffusers_functionality():
    """测试Diffusers功能"""
    print("\n🔧 测试Diffusers功能...")
    
    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        import torch
        
        print("✅ Diffusers模块导入成功")
        
        # 创建小模型测试
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            sample_size=32,
        )
        
        print("✅ VAE模型创建成功")
        
        # 测试前向传播
        with torch.no_grad():
            test_input = torch.randn(1, 3, 32, 32)
            latents = vae.encode(test_input).latent_dist.sample()
            reconstructed = vae.decode(latents).sample
            
            print(f"✅ VAE前向传播成功: {test_input.shape} -> {latents.shape} -> {reconstructed.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Diffusers功能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 Kaggle版本修复工具")
    print("=" * 50)
    
    # 1. 检查可用版本
    check_available_versions()
    
    # 2. 安装兼容版本
    install_compatible_versions()
    
    # 3. 测试导入
    print("\n" + "=" * 30 + " 测试阶段 " + "=" * 30)
    
    if test_imports():
        print("✅ 所有包导入成功")
        
        # 4. 测试Diffusers功能
        if test_diffusers_functionality():
            print("✅ Diffusers功能测试通过")
            print("\n🎉 修复完成！可以开始训练")
        else:
            print("⚠️  Diffusers功能测试失败，但基本导入正常")
    else:
        print("❌ 部分包导入失败")
    
    print("\n📋 下一步:")
    print("1. 运行: python train_kaggle.py --stage all")
    print("2. 或者分步运行训练脚本")

if __name__ == "__main__":
    main()
