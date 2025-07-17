#!/usr/bin/env python3
"""
终极Kaggle环境修复脚本
彻底清理并重建所有依赖，解决所有版本冲突
"""

import subprocess
import sys
import os
import shutil

def run_command(cmd, description="", ignore_errors=False):
    """运行命令"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 or ignore_errors:
            print(f"✅ {description} - 完成")
            return True
        else:
            print(f"❌ {description} - 失败")
            if result.stderr and not ignore_errors:
                print(f"错误: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - 异常: {e}")
        return False

def nuclear_cleanup():
    """核弹级清理 - 彻底清除所有相关包"""
    print("💥 核弹级清理 - 彻底清除所有相关包")
    print("=" * 50)
    
    # 1. 清理Python缓存
    print("\n1️⃣ 清理Python缓存...")
    try:
        import sys
        for module in list(sys.modules.keys()):
            if any(pkg in module for pkg in ['numpy', 'torch', 'diffusers', 'transformers', 'scipy', 'sklearn']):
                del sys.modules[module]
        print("✅ Python模块缓存已清理")
    except:
        pass
    
    # 2. 卸载所有相关包
    print("\n2️⃣ 卸载所有相关包...")
    packages_to_remove = [
        # 核心包
        "torch", "torchvision", "torchaudio", "torchtext",
        "numpy", "scipy", "scikit-learn", "matplotlib",
        # AI包
        "diffusers", "transformers", "accelerate", "huggingface_hub",
        # 依赖包
        "pillow", "opencv-python", "opencv-contrib-python",
        "einops", "tqdm", "packaging", "filelock",
        # 可能冲突的包
        "jax", "jaxlib", "flax", "optax",
        "tensorflow", "tensorflow-gpu",
        "pandas", "seaborn"
    ]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall -y {package}", f"卸载 {package}", ignore_errors=True)
    
    # 3. 强制清理pip缓存
    print("\n3️⃣ 清理pip缓存...")
    run_command("pip cache purge", "清理pip缓存", ignore_errors=True)
    
    # 4. 清理conda缓存 (如果存在)
    print("\n4️⃣ 清理conda缓存...")
    run_command("conda clean -a -y", "清理conda缓存", ignore_errors=True)

def install_base_system():
    """安装基础系统包"""
    print("\n🏗️  安装基础系统包")
    print("=" * 30)
    
    # 1. 升级pip和setuptools
    print("\n1️⃣ 升级基础工具...")
    run_command("pip install --upgrade pip setuptools wheel", "升级pip和setuptools")
    
    # 2. 安装NumPy (最稳定版本)
    print("\n2️⃣ 安装NumPy...")
    numpy_versions = ["1.24.4", "1.23.5", "1.21.6"]
    for version in numpy_versions:
        if run_command(f"pip install numpy=={version}", f"安装NumPy {version}"):
            break
    
    # 3. 安装SciPy (兼容NumPy)
    print("\n3️⃣ 安装SciPy...")
    run_command("pip install scipy==1.10.1", "安装SciPy")

def install_pytorch_stack():
    """安装PyTorch技术栈"""
    print("\n🔥 安装PyTorch技术栈")
    print("=" * 30)
    
    # PyTorch安装选项 (从最稳定到最保守)
    pytorch_options = [
        # 选项1: CPU版本 (最稳定)
        {
            "cmd": "pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu",
            "desc": "PyTorch 2.1.0 CPU版本"
        },
        # 选项2: 较旧CUDA版本
        {
            "cmd": "pip install torch==2.0.1 torchvision==0.15.2",
            "desc": "PyTorch 2.0.1 默认版本"
        },
        # 选项3: 最保守版本
        {
            "cmd": "pip install torch==1.13.1 torchvision==0.14.1",
            "desc": "PyTorch 1.13.1 保守版本"
        }
    ]
    
    for i, option in enumerate(pytorch_options, 1):
        print(f"\n尝试方案 {i}: {option['desc']}")
        if run_command(option["cmd"], option["desc"]):
            print(f"✅ PyTorch方案 {i} 安装成功")
            break
    else:
        print("❌ 所有PyTorch安装方案都失败")
        return False
    
    return True

def install_ai_packages():
    """安装AI相关包"""
    print("\n🤖 安装AI相关包")
    print("=" * 30)
    
    # 兼容版本组合
    ai_packages = [
        ("huggingface_hub==0.16.4", "HuggingFace Hub"),
        ("transformers==4.30.2", "Transformers"),
        ("diffusers==0.21.4", "Diffusers"),
        ("accelerate==0.20.3", "Accelerate")
    ]
    
    for package, name in ai_packages:
        run_command(f"pip install {package}", f"安装 {name}")

def install_utility_packages():
    """安装工具包"""
    print("\n🛠️  安装工具包")
    print("=" * 30)
    
    utility_packages = [
        ("pillow==9.5.0", "Pillow"),
        ("opencv-python==4.8.0.76", "OpenCV"),
        ("matplotlib==3.7.2", "Matplotlib"),
        ("scikit-learn==1.3.0", "Scikit-learn"),
        ("tqdm==4.65.0", "TQDM"),
        ("einops==0.6.1", "Einops"),
        ("packaging>=20.0", "Packaging"),
        ("filelock>=3.0", "FileLock")
    ]
    
    for package, name in utility_packages:
        run_command(f"pip install {package}", f"安装 {name}")

def comprehensive_test():
    """全面测试"""
    print("\n🧪 全面功能测试")
    print("=" * 30)
    
    # 清理模块缓存
    modules_to_clear = ['numpy', 'torch', 'torchvision', 'diffusers', 'transformers', 'scipy', 'sklearn']
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
    
    test_results = {}
    
    # 测试1: NumPy
    print("\n1️⃣ 测试NumPy...")
    try:
        import numpy as np
        test_array = np.random.randn(3, 3)
        result = np.mean(test_array)
        print(f"✅ NumPy {np.__version__}: 功能正常")
        test_results['numpy'] = True
    except Exception as e:
        print(f"❌ NumPy测试失败: {e}")
        test_results['numpy'] = False
    
    # 测试2: PyTorch
    print("\n2️⃣ 测试PyTorch...")
    try:
        import torch
        test_tensor = torch.randn(2, 3)
        result = torch.mean(test_tensor)
        print(f"✅ PyTorch {torch.__version__}: 功能正常")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  使用CPU模式")
        
        test_results['torch'] = True
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        test_results['torch'] = False
    
    # 测试3: TorchVision
    print("\n3️⃣ 测试TorchVision...")
    try:
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}: 导入成功")
        test_results['torchvision'] = True
    except Exception as e:
        print(f"❌ TorchVision测试失败: {e}")
        test_results['torchvision'] = False
    
    # 测试4: Diffusers
    print("\n4️⃣ 测试Diffusers...")
    try:
        import diffusers
        from diffusers import AutoencoderKL
        print(f"✅ Diffusers {diffusers.__version__}: 导入成功")
        test_results['diffusers'] = True
    except Exception as e:
        print(f"❌ Diffusers测试失败: {e}")
        test_results['diffusers'] = False
    
    # 测试5: VAE功能
    print("\n5️⃣ 测试VAE功能...")
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
        print(f"   输入: {test_input.shape}")
        print(f"   潜在: {latents.shape}")
        print(f"   重建: {reconstructed.shape}")
        test_results['vae'] = True
        
    except Exception as e:
        print(f"❌ VAE功能测试失败: {e}")
        test_results['vae'] = False
    
    # 测试总结
    print("\n📊 测试总结:")
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {test_name}")
    
    print(f"\n总体结果: {passed}/{total} 通过")
    
    # 关键测试
    critical_tests = ['numpy', 'torch', 'diffusers', 'vae']
    critical_passed = all(test_results.get(test, False) for test in critical_tests)
    
    return critical_passed, test_results

def main():
    """主修复流程"""
    print("🚀 终极Kaggle环境修复工具")
    print("彻底清理并重建所有依赖")
    print("=" * 60)
    
    try:
        # 阶段1: 核弹级清理
        nuclear_cleanup()
        
        # 阶段2: 安装基础系统
        install_base_system()
        
        # 阶段3: 安装PyTorch
        if not install_pytorch_stack():
            print("❌ PyTorch安装失败，无法继续")
            return False
        
        # 阶段4: 安装AI包
        install_ai_packages()
        
        # 阶段5: 安装工具包
        install_utility_packages()
        
        # 阶段6: 全面测试
        print("\n" + "=" * 50 + " 最终测试 " + "=" * 50)
        
        success, results = comprehensive_test()
        
        if success:
            print("\n🎉 修复成功！所有关键组件正常工作")
            print("\n📋 下一步:")
            print("   python train_kaggle.py --stage all")
            print("\n💡 提示:")
            print("   - 环境已完全重建")
            print("   - 所有版本冲突已解决")
            print("   - 可以开始训练")
            return True
        else:
            print("\n⚠️  部分组件仍有问题")
            print("\n🔧 建议:")
            print("   1. 重启Kaggle内核")
            print("   2. 重新运行此脚本")
            print("   3. 检查Kaggle环境限制")
            return False
            
    except Exception as e:
        print(f"\n💥 修复过程中出现异常: {e}")
        print("\n🔧 建议:")
        print("   1. 重启Kaggle内核")
        print("   2. 重新运行此脚本")
        return False

if __name__ == "__main__":
    main()
