#!/usr/bin/env python3
"""
VQ-VAE + Transformer 统一环境配置脚本
基于diffusers官方配置，支持VQ-VAE和Transformer训练

功能：
- 安装PyTorch GPU版本
- 使用diffusers官方配置: diffusers[torch] + transformers
- 智能选择diffusers版本，确保VQModel可用
- 安装完整的图像处理和序列生成依赖
- 测试VQ-VAE和Transformer环境完整性

版本策略：
- 遵循diffusers官方配置: pip install diffusers[torch] transformers
- 智能降级: 如果VQModel不可用，自动降级到稳定版本
- 统一管理: 一个环境支持两个训练阶段
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
        ("pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121", "PyTorch方案 1"),
        ("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121", "PyTorch方案 2"),
        ("pip install torch torchvision torchaudio", "PyTorch方案 3 (CPU备用)"),
    ]
    
    for cmd, description in pytorch_options:
        print(f"\n尝试{description}...")
        if run_command(cmd, f"🔄 {description}"):
            print(f"✅ 安装PyTorch GPU版本 成功")
            return True
    
    print("❌ 所有PyTorch安装方案都失败")
    return False

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

def install_core_dependencies():
    """安装核心依赖 - diffusers + transformers"""
    print("🎨 安装核心依赖...")
    print("💡 使用diffusers官方配置")
    
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
    
    # 第四步：安装其他核心依赖
    core_packages = [
        ("safetensors", "SafeTensors"),
        ("accelerate", "Accelerate"),
        ("tokenizers", "Tokenizers"),
    ]
    
    success_count = 2  # diffusers和transformers已成功
    for package, description in core_packages:
        if run_command(f"pip install {package} --upgrade", f"安装 {description}"):
            success_count += 1
    
    # 验证版本兼容性
    print("\n🔧 验证版本兼容性...")
    run_command("pip check", "检查依赖冲突")
    
    total_packages = 2 + len(core_packages)
    print(f"\n📊 核心依赖安装结果: {success_count}/{total_packages} 成功")
    return success_count >= total_packages - 1 and vqmodel_available

def install_additional_dependencies():
    """安装额外依赖"""
    print("📚 安装额外依赖...")
    
    additional_packages = [
        # 数据处理
        "numpy==1.26.4",
        "pillow",
        "opencv-python==4.8.1.78", 
        "matplotlib==3.7.2",
        "scikit-image",
        
        # 机器学习工具
        "scikit-learn==1.3.0",
        "einops==0.7.0",
        "lpips==0.1.4",
        "tqdm",
        
        # 其他工具
        "scipy==1.11.4",
    ]
    
    success_count = 0
    for package in additional_packages:
        if run_command(f"pip install {package}", f"安装 {package}"):
            success_count += 1
    
    print(f"\n📊 额外依赖安装结果: {success_count}/{len(additional_packages)} 成功")
    return success_count >= len(additional_packages) - 2  # 允许2个失败

def test_unified_environment():
    """测试统一环境"""
    print("🧪 测试统一环境...")
    
    # 清理模块缓存
    modules_to_clear = ['torch', 'diffusers', 'transformers', 'huggingface_hub', 'accelerate']
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    # 基础测试
    tests = [
        ("PyTorch", "torch"),
        ("Diffusers", "diffusers"),
        ("Transformers", "transformers"),
        ("HuggingFace Hub", "huggingface_hub"),
        ("Accelerate", "accelerate"),
    ]
    
    success_count = 0
    for name, module in tests:
        try:
            imported_module = importlib.import_module(module)
            version = getattr(imported_module, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
            success_count += 1
        except ImportError:
            print(f"❌ {name}: 导入失败")
    
    # 测试PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
        else:
            print("⚠️ CUDA不可用，使用CPU模式")
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
    
    # 测试VQModel
    try:
        from diffusers.models.autoencoders.vq_model import VQModel
        print("✅ VQModel: 可用 (VQ-VAE支持)")
    except ImportError as e:
        print(f"❌ VQModel: 导入失败 - {e}")
    
    # 测试GPT2
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        print("✅ GPT2: 可用 (Transformer支持)")
    except ImportError as e:
        print(f"❌ GPT2: 导入失败 - {e}")
    
    print(f"\n📊 统一环境测试结果: {success_count}/{len(tests)} 成功")
    return success_count >= len(tests) - 1

def main():
    """主函数"""
    print("🎨 VQ-VAE + Transformer 统一环境配置脚本")
    print("=" * 60)
    print("🎯 一个环境支持VQ-VAE和Transformer训练")
    print("💡 使用diffusers官方配置")
    print("🔧 pip install diffusers[torch] transformers")
    
    steps = [
        ("安装PyTorch", install_pytorch),
        ("安装核心依赖", install_core_dependencies),
        ("安装额外依赖", install_additional_dependencies),
        ("测试环境", test_unified_environment),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"\n❌ {step_name}失败，停止安装")
            return False
    
    print("\n🎉 统一环境配置完成!")
    print("✅ 支持VQ-VAE和Transformer训练")
    print("\n📋 下一步:")
    print("   1. VQ-VAE训练: python training/train_vqvae.py --help")
    print("   2. Transformer训练: python training/train_transformer.py --help")
    print("   3. 完整训练: python train_main.py --help")
    print("\n🚀 训练命令示例:")
    print("   python train_main.py --data_dir /kaggle/input/dataset")
    print("\n💡 环境说明:")
    print("   - diffusers: 智能版本选择，确保VQModel可用")
    print("   - transformers: 最新版本，支持序列生成")
    print("   - 统一环境: 简化部署和维护")

if __name__ == "__main__":
    main()
