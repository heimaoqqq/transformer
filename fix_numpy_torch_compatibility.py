#!/usr/bin/env python3
"""
专门修复NumPy 2.x和PyTorch兼容性问题
解决 'torch.sparse._triton_ops_meta' 和 NumPy版本冲突
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """运行命令"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - 成功")
            return True
        else:
            print(f"❌ {description} - 失败")
            if result.stderr:
                print(f"错误: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - 异常: {e}")
        return False

def check_current_environment():
    """检查当前环境"""
    print("🔍 检查当前环境...")
    
    # 检查NumPy版本
    try:
        import numpy as np
        print(f"📦 NumPy版本: {np.__version__}")
        if np.__version__.startswith('2.'):
            print("⚠️  NumPy 2.x 检测到，可能导致兼容性问题")
            return False
        else:
            print("✅ NumPy版本兼容")
    except ImportError:
        print("❌ NumPy未安装")
        return False
    
    # 检查PyTorch
    try:
        import torch
        print(f"📦 PyTorch版本: {torch.__version__}")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA不可用")
            
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    except Exception as e:
        print(f"⚠️  PyTorch导入异常: {e}")
        return False
    
    return True

def fix_numpy_compatibility():
    """修复NumPy兼容性"""
    print("\n🔧 修复NumPy兼容性...")
    
    # 1. 卸载NumPy 2.x
    print("\n1️⃣ 卸载NumPy 2.x...")
    run_command("pip uninstall -y numpy", "卸载NumPy")
    
    # 2. 安装NumPy 1.x
    print("\n2️⃣ 安装兼容的NumPy 1.x...")
    numpy_versions = [
        "1.24.4",  # 稳定版本
        "1.24.3",  # 备选
        "1.23.5",  # 保守版本
        "1.21.6"   # 最保守
    ]
    
    for version in numpy_versions:
        if run_command(f"pip install numpy=={version}", f"安装NumPy {version}"):
            print(f"✅ NumPy {version} 安装成功")
            break
    else:
        print("⚠️  所有NumPy版本都失败")
        return False
    
    return True

def fix_pytorch_triton():
    """修复PyTorch Triton问题"""
    print("\n🔧 修复PyTorch Triton问题...")
    
    # 1. 重新安装PyTorch (CPU版本，避免Triton问题)
    print("\n1️⃣ 重新安装PyTorch...")
    
    # 卸载现有PyTorch
    run_command("pip uninstall -y torch torchvision torchaudio", "卸载PyTorch相关包")
    
    # 安装CPU版本 (避免Triton依赖)
    pytorch_options = [
        # 选项1: CPU版本
        "pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu",
        # 选项2: CUDA版本但指定兼容版本
        "pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118",
        # 选项3: 默认版本
        "pip install torch==1.13.1 torchvision==0.14.1"
    ]
    
    for i, cmd in enumerate(pytorch_options, 1):
        print(f"\n尝试方案 {i}...")
        if run_command(cmd, f"PyTorch安装方案 {i}"):
            print(f"✅ PyTorch方案 {i} 成功")
            break
    else:
        print("⚠️  所有PyTorch安装方案都失败")
        return False
    
    return True

def reinstall_dependencies():
    """重新安装依赖"""
    print("\n📦 重新安装依赖...")
    
    # 核心依赖列表 (兼容版本)
    dependencies = [
        "scipy==1.10.1",
        "scikit-learn==1.3.0", 
        "matplotlib==3.7.2",
        "pillow==9.5.0",
        "opencv-python==4.8.0.76",
        "tqdm==4.65.0",
        "einops==0.6.1"
    ]
    
    for dep in dependencies:
        run_command(f"pip install {dep}", f"安装 {dep}")

def install_diffusers_compatible():
    """安装兼容的Diffusers"""
    print("\n🎨 安装兼容的Diffusers...")
    
    # 1. 先安装兼容的HuggingFace Hub
    run_command("pip install huggingface_hub==0.16.4", "安装兼容的HuggingFace Hub")
    
    # 2. 安装较旧但稳定的Diffusers
    diffusers_versions = [
        "0.21.4",  # 较旧但稳定
        "0.20.2",  # 更保守
        "0.19.3"   # 最保守
    ]
    
    for version in diffusers_versions:
        if run_command(f"pip install diffusers=={version}", f"安装Diffusers {version}"):
            print(f"✅ Diffusers {version} 安装成功")
            break
    
    # 3. 安装兼容的Transformers
    run_command("pip install transformers==4.30.2", "安装兼容的Transformers")
    
    # 4. 安装兼容的Accelerate
    run_command("pip install accelerate==0.20.3", "安装兼容的Accelerate")

def test_final_compatibility():
    """最终兼容性测试"""
    print("\n🧪 最终兼容性测试...")
    
    # 清理模块缓存
    modules_to_clear = ['numpy', 'torch', 'torchvision', 'diffusers', 'transformers']
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
    
    success_count = 0
    total_tests = 5
    
    # 测试NumPy
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        # 测试基本功能
        test_array = np.random.randn(3, 3)
        result = np.mean(test_array)
        print(f"✅ NumPy功能正常: mean={result:.4f}")
        success_count += 1
    except Exception as e:
        print(f"❌ NumPy测试失败: {e}")
    
    # 测试PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # 测试基本功能
        test_tensor = torch.randn(2, 3)
        result = torch.mean(test_tensor)
        print(f"✅ PyTorch功能正常: mean={result:.4f}")
        success_count += 1
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
    
    # 测试TorchVision
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
        success_count += 1
    except Exception as e:
        print(f"❌ TorchVision测试失败: {e}")
    
    # 测试Diffusers
    try:
        import diffusers
        print(f"✅ Diffusers: {diffusers.__version__}")
        success_count += 1
    except Exception as e:
        print(f"❌ Diffusers测试失败: {e}")
    
    # 测试VAE功能
    try:
        from diffusers import AutoencoderKL
        import torch
        
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            sample_size=32,
        )
        
        with torch.no_grad():
            test_input = torch.randn(1, 3, 32, 32)
            latents = vae.encode(test_input).latent_dist.sample()
            reconstructed = vae.decode(latents).sample
        
        print("✅ VAE功能测试通过")
        success_count += 1
        
    except Exception as e:
        print(f"❌ VAE功能测试失败: {e}")
    
    print(f"\n📊 测试结果: {success_count}/{total_tests}")
    return success_count == total_tests

def main():
    """主修复流程"""
    print("🔧 NumPy & PyTorch 兼容性修复工具")
    print("解决NumPy 2.x和Triton模块问题")
    print("=" * 60)
    
    # 1. 检查当前环境
    if check_current_environment():
        print("✅ 当前环境基本正常")
    else:
        print("⚠️  检测到兼容性问题，开始修复...")
    
    # 2. 修复NumPy
    if not fix_numpy_compatibility():
        print("❌ NumPy修复失败")
        return False
    
    # 3. 修复PyTorch
    if not fix_pytorch_triton():
        print("❌ PyTorch修复失败")
        return False
    
    # 4. 重新安装依赖
    reinstall_dependencies()
    
    # 5. 安装兼容的Diffusers
    install_diffusers_compatible()
    
    # 6. 最终测试
    print("\n" + "=" * 40 + " 最终测试 " + "=" * 40)
    
    if test_final_compatibility():
        print("\n🎉 修复成功！所有组件正常工作")
        print("\n📋 下一步:")
        print("   python train_kaggle.py --stage all")
        return True
    else:
        print("\n⚠️  部分组件仍有问题")
        print("\n🔧 建议:")
        print("   1. 重启Kaggle内核")
        print("   2. 重新运行此脚本")
        print("   3. 或使用CPU版本训练")
        return False

if __name__ == "__main__":
    main()
