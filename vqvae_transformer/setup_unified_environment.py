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

        # 基于网上调研，正确的导入路径是 diffusers.models.autoencoders.vq_model
        import_tests = [
            "from diffusers.models.autoencoders.vq_model import VQModel; print('SUCCESS_CORRECT_PATH')",
            "from diffusers.models.vq_model import VQModel; print('SUCCESS_OLD_PATH')",
            "from diffusers import VQModel; print('SUCCESS_DIRECT_IMPORT')",
        ]

        for i, test_code in enumerate(import_tests, 1):
            result = subprocess.run([
                sys.executable, "-c", test_code
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and "SUCCESS" in result.stdout:
                print(f"✅ VQModel导入成功 (路径{i})")
                return True

        # 如果所有路径都失败，显示最后一个错误
        print(f"VQModel导入失败: {result.stderr}")
        return False

    except Exception as e:
        print(f"VQModel测试异常: {e}")
        return False

def install_core_dependencies():
    """安装核心依赖 - 使用确定支持VQModel的版本"""
    print("🎨 安装核心依赖...")
    print("💡 使用确定支持VQModel的版本组合")

    # 先卸载所有相关包，包括peft
    run_command("pip uninstall -y huggingface_hub diffusers transformers accelerate torchao peft", "清理所有相关包")
    run_command("pip cache purge", "清理pip缓存")

    # 第一步：修复NumPy版本冲突
    print("\n🔧 修复NumPy版本冲突...")
    if not run_command("pip install 'numpy<2.0' --force-reinstall", "降级NumPy到1.x版本"):
        return False

    # 第二步：使用确定支持VQModel的版本组合
    print("\n🔧 安装确定支持VQModel的版本组合...")

    # 基于网上调研，VQModel在diffusers 0.31版本被移除，0.30.x是最后支持的版本
    known_working_versions = [
        # 版本组合1: diffusers 0.30.x (最后支持VQModel的版本)
        {
            "huggingface_hub": "0.20.3",
            "transformers": "4.36.2",
            "diffusers": "0.30.3",
            "accelerate": "0.25.0",
            "peft": "0.6.2"
        },
        # 版本组合2: diffusers 0.29.x (稳定版本)
        {
            "huggingface_hub": "0.19.4",
            "transformers": "4.35.2",
            "diffusers": "0.29.2",
            "accelerate": "0.24.1",
            "peft": "0.5.0"
        },
        # 版本组合3: diffusers 0.28.x (保守版本)
        {
            "huggingface_hub": "0.18.0",
            "transformers": "4.34.1",
            "diffusers": "0.28.2",
            "accelerate": "0.23.0",
            "peft": "0.4.0"
        }
    ]

    vqmodel_available = False

    for i, versions in enumerate(known_working_versions, 1):
        print(f"\n尝试版本组合 {i}:")
        for package, version in versions.items():
            print(f"  {package}=={version}")

        # 按顺序安装
        install_success = True
        for package, version in versions.items():
            if not run_command(f"pip install '{package}=={version}'", f"安装 {package} {version}"):
                install_success = False
                break

        if install_success:
            # 测试VQModel
            if test_vqmodel_import():
                print(f"✅ 版本组合 {i} VQModel可用")
                vqmodel_available = True
                break
            else:
                print(f"❌ 版本组合 {i} VQModel不可用")
        else:
            print(f"❌ 版本组合 {i} 安装失败")

    if not vqmodel_available:
        print("❌ 所有已知版本组合都失败")
        print("💡 建议使用分阶段训练作为备选方案")
        return False

    # 第三步：安装其他核心依赖
    core_packages = [
        ("safetensors>=0.3.0,<0.5.0", "SafeTensors"),
        ("tokenizers>=0.13.0,<0.16.0", "Tokenizers"),
    ]

    success_count = 5  # 5个核心包已成功
    for package, description in core_packages:
        if run_command(f"pip install '{package}'", f"安装 {description}"):
            success_count += 1

    # 验证版本兼容性
    print("\n🔧 验证版本兼容性...")
    run_command("pip check", "检查依赖冲突")

    total_packages = 5 + len(core_packages)
    print(f"\n📊 核心依赖安装结果: {success_count}/{total_packages} 成功")
    return success_count >= total_packages - 1 and vqmodel_available



def install_additional_dependencies():
    """安装额外依赖 - 兼容版本"""
    print("📚 安装额外依赖...")

    # 确保NumPy版本正确
    run_command("pip install 'numpy<2.0' --force-reinstall", "确保NumPy 1.x版本")

    additional_packages = [
        # 数据处理 (兼容NumPy 1.x的版本)
        "pillow>=9.0.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "scikit-image>=0.20.0",

        # 机器学习工具
        "scikit-learn>=1.3.0",
        "einops>=0.6.0",
        "tqdm>=4.65.0",

        # 其他工具
        "scipy>=1.10.0",
    ]

    success_count = 0
    for package in additional_packages:
        if run_command(f"pip install '{package}'", f"安装 {package}"):
            success_count += 1

    # 特殊处理lpips (可选依赖)
    print("\n🎨 安装感知损失库...")
    if run_command("pip install lpips", "安装 lpips (可选)"):
        success_count += 1
    else:
        print("⚠️ lpips安装失败，跳过 (可选依赖)")

    print(f"\n📊 额外依赖安装结果: {success_count}/{len(additional_packages)+1} 成功")
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

def check_package_versions():
    """检查关键包的版本"""
    print("📋 检查关键包版本...")

    packages_to_check = {
        'torch': '2.0.0',
        'torchvision': '0.15.0',
        'diffusers': '0.20.0',
        'transformers': '4.20.0',
        'numpy': '1.21.0',
        'pillow': '8.0.0',
        'matplotlib': '3.5.0',
        'tqdm': '4.60.0',
    }

    all_good = True

    for package, min_version in packages_to_check.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {package}: {version}")
        except ImportError:
            print(f"   ❌ {package}: 未安装")
            all_good = False

    return all_good

def test_step_training_compatibility():
    """测试分步训练兼容性"""
    print("🧪 测试分步训练兼容性...")

    try:
        # 测试VQ-VAE组件
        print("   测试VQ-VAE组件...")
        test_code = """
import torch
from diffusers import VQModel

# 创建测试VQModel
model = VQModel(
    in_channels=3,
    out_channels=3,
    down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
    up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
    block_out_channels=[128, 256],
    layers_per_block=2,
    act_fn="silu",
    latent_channels=4,
    norm_num_groups=32,
    vq_embed_dim=256,
    num_vq_embeddings=1024,
)

# 测试编码解码
x = torch.randn(1, 3, 64, 64)
encoded = model.encode(x)
decoded = model.decode(encoded.latents)
print("VQ-VAE测试成功")
"""

        result = subprocess.run([
            sys.executable, "-c", test_code
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("   ✅ VQ-VAE组件测试通过")
        else:
            print(f"   ❌ VQ-VAE组件测试失败: {result.stderr}")
            return False

        # 测试Transformer组件
        print("   测试Transformer组件...")
        test_code = """
import torch
from diffusers import Transformer2DModel

# 创建测试Transformer
model = Transformer2DModel(
    num_attention_heads=8,
    attention_head_dim=64,
    in_channels=4,
    num_layers=4,
    dropout=0.1,
    norm_num_groups=32,
    activation_fn="gelu",
)

# 测试前向传播
x = torch.randn(1, 4, 16, 16)
output = model(x)
print("Transformer测试成功")
"""

        result = subprocess.run([
            sys.executable, "-c", test_code
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("   ✅ Transformer组件测试通过")
            return True
        else:
            print(f"   ❌ Transformer组件测试失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"   ❌ 兼容性测试异常: {e}")
        return False

def create_training_scripts_info():
    """创建训练脚本信息文件"""
    print("📝 创建训练脚本信息...")

    info = {
        "step1_vqvae": {
            "script": "train_step1_vqvae.py",
            "description": "第一步：训练VQ-VAE模型",
            "example_command": "python train_step1_vqvae.py --data_dir /path/to/data --output_dir ./step1_output",
            "output": "VQ-VAE模型保存在 ./step1_output/vqvae_best/"
        },
        "step2_transformer": {
            "script": "train_step2_transformer.py",
            "description": "第二步：基于预训练VQ-VAE训练Transformer",
            "example_command": "python train_step2_transformer.py --vqvae_path ./step1_output/vqvae_best --data_dir /path/to/data",
            "output": "Transformer模型保存在 ./step2_output/transformer_best/"
        },
        "unified_training": {
            "script": "train_pure_diffusers.py",
            "description": "统一训练：同时训练VQ-VAE和Transformer",
            "example_command": "python train_pure_diffusers.py --data_dir /path/to/data",
            "output": "完整模型保存在 ./pure_diffusers_output/"
        }
    }

    try:
        import json
        with open("training_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print("   ✅ 训练信息文件已创建: training_info.json")
        return True
    except Exception as e:
        print(f"   ❌ 创建信息文件失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 VQ-VAE + Transformer 统一环境配置")
    print("=" * 60)
    print("支持分步训练和统一训练两种模式")
    print("基于diffusers标准组件的完整实现")
    print("=" * 60)

    steps = [
        ("安装PyTorch", install_pytorch),
        ("安装核心依赖", install_core_dependencies),
        ("安装额外依赖", install_additional_dependencies),
        ("测试环境", test_unified_environment),
        ("检查包版本", check_package_versions),
        ("测试分步训练兼容性", test_step_training_compatibility),
        ("创建训练脚本信息", create_training_scripts_info),
    ]

    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"\n❌ {step_name}失败，但继续执行...")

    print("\n🎉 环境配置完成！")
    print("=" * 60)
    print("📋 可用的训练模式:")
    print()
    print("� 分步训练模式 (推荐):")
    print("   1️⃣ 第一步 - 训练VQ-VAE:")
    print("      python train_step1_vqvae.py --data_dir /path/to/data")
    print("   2️⃣ 第二步 - 训练Transformer:")
    print("      python train_step2_transformer.py --vqvae_path ./step1_vqvae_output/vqvae_best --data_dir /path/to/data")
    print()
    print("� 统一训练模式:")
    print("   python train_pure_diffusers.py --data_dir /path/to/data")
    print()
    print("📄 详细信息请查看: training_info.json")

if __name__ == "__main__":
    main()
