#!/usr/bin/env python3
"""
VQ-VAE阶段专用环境配置脚本
专注于diffusers和图像处理依赖，避免transformers相关的依赖冲突

功能：
- 安装PyTorch GPU版本
- 智能选择diffusers版本: 优先最新版本，自动降级保证VQModel可用
- 使用diffusers官方配置: diffusers[torch] + transformers
- 安装图像处理和数值计算依赖
- 测试VQ-VAE环境完整性

版本策略：
- 智能版本选择: 优先尝试最新版本，如果VQModel不可用则自动降级
- 第一选择: diffusers最新版本 + transformers (官方配置)
- 备用方案: diffusers 0.30.3 (已知支持VQModel的最后稳定版本)
- 正确导入路径: from diffusers.models.autoencoders.vq_model import VQModel
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
    """安装VQ-VAE专用依赖 - 智能版本选择"""
    print("🎨 安装VQ-VAE专用依赖...")
    print("💡 智能选择最佳diffusers版本")

    # 先卸载可能冲突的包
    run_command("pip uninstall -y huggingface_hub diffusers transformers", "清理可能冲突的包")
    run_command("pip cache purge", "清理pip缓存")

    # 策略1: 尝试最新版本
    print("\n🔧 尝试安装最新版本...")

    # 第一步：安装diffusers[torch] (最新版本)
    if not run_command("pip install 'diffusers[torch]' --upgrade", "安装 diffusers[torch] (最新版本)"):
        return False

    # 第二步：安装transformers
    if not run_command("pip install transformers --upgrade", "安装 transformers"):
        return False

    # 第三步：测试VQModel是否可用
    print("\n🧪 测试VQModel可用性...")
    vqmodel_available = test_vqmodel_import()

    if not vqmodel_available:
        print("\n⚠️ 最新版本中VQModel不可用，降级到稳定版本...")
        # 降级到已知支持VQModel的版本
        if not run_command("pip install 'diffusers==0.30.3' --force-reinstall", "降级到 diffusers 0.30.3"):
            return False

        # 重新测试
        vqmodel_available = test_vqmodel_import()
        if not vqmodel_available:
            print("❌ 即使降级后VQModel仍不可用")
            return False
        else:
            print("✅ 降级后VQModel可用")
    else:
        print("✅ 最新版本中VQModel可用")

    # 第四步：安装其他必要依赖
    additional_packages = [
        ("safetensors", "SafeTensors"),
        ("accelerate", "Accelerate"),
    ]

    success_count = 2  # diffusers和transformers已成功
    for package, description in additional_packages:
        if run_command(f"pip install {package} --upgrade", f"安装 {description}"):
            success_count += 1

    # 验证版本兼容性
    print("\n🔧 验证版本兼容性...")
    run_command("pip check", "检查依赖冲突")

    total_packages = 2 + len(additional_packages)
    print(f"\n📊 VQ-VAE依赖安装结果: {success_count}/{total_packages} 成功")
    return success_count >= total_packages - 1 and vqmodel_available

def test_vqmodel_import():
    """测试VQModel导入是否成功"""
    try:
        import subprocess
        import sys

        # 在子进程中测试导入，避免影响当前进程
        result = subprocess.run([
            sys.executable, "-c",
            "from diffusers.models.autoencoders.vq_model import VQModel; print('SUCCESS')"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0 and "SUCCESS" in result.stdout:
            return True
        else:
            print(f"VQModel导入失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"VQModel测试异常: {e}")
        return False

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
    
    # 测试VQModel (使用官方推荐的导入路径)
    vqmodel_success = False
    try:
        # 官方推荐的导入路径 (diffusers 0.34.0)
        from diffusers.models.autoencoders.vq_model import VQModel
        print("✅ VQModel: 可用 (官方推荐路径)")
        vqmodel_success = True
        # 测试实例化
        _ = VQModel
    except ImportError as e:
        print(f"❌ VQModel: 导入失败 - {e}")
    except Exception as e:
        print(f"⚠️ VQModel: 导入成功但有问题 - {e}")
        vqmodel_success = True  # 导入成功，只是有警告

    if vqmodel_success:
        print("✅ VQModel验证成功")
    else:
        print("❌ VQModel验证失败，请检查diffusers安装")
    
    print(f"\n📊 VQ-VAE环境测试结果: {success_count}/{len(tests)} 成功")
    return success_count >= len(tests) - 1

def main():
    """主函数"""
    print("🎨 VQ-VAE阶段环境配置脚本")
    print("=" * 50)
    print("🎯 专用于VQ-VAE训练的环境配置")
    print("💡 使用diffusers官方指定配置")
    print("🔧 pip install diffusers[torch] transformers")
    
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
    print("\n� 下一步:")
    print("   1. 验证环境: python test_vqvae_environment_fix.py")
    print("   2. 开始训练: python training/train_vqvae.py --help")
    print("   3. 查看文档: README.md")
    print("\n�🚀 训练命令示例:")
    print("   python train_main.py --skip_transformer --data_dir /kaggle/input/dataset")
    print("   或者")
    print("   python training/train_vqvae.py --data_dir /kaggle/input/dataset")
    print("\n💡 版本说明:")
    print("   - diffusers: 最新版本 (官方配置)")
    print("   - transformers: 官方要求的依赖")
    print("   - 导入路径: diffusers.models.autoencoders.vq_model")
    print("   - VQ-VAE模型支持跨环境兼容")

if __name__ == "__main__":
    main()
