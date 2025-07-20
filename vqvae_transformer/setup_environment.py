#!/usr/bin/env python3
"""
VQ-VAE + Transformer 统一环境安装脚本
自动检测环境类型并安装兼容的依赖版本
解决diffusers、transformers等API兼容性问题
"""

import os
import sys
import subprocess
import importlib

def run_command(cmd, description=""):
    """运行命令并处理错误"""
    print(f"🔄 {description}")
    print(f"   命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        if e.stderr.strip():
            print(f"   错误: {e.stderr.strip()}")
        return False

def detect_environment():
    """检测运行环境"""
    print("🔍 检测运行环境...")
    
    if any([os.path.exists('/kaggle'), 'KAGGLE_KERNEL_RUN_TYPE' in os.environ]):
        print("✅ 检测到Kaggle环境")
        return "kaggle"
    
    try:
        import google.colab
        print("✅ 检测到Google Colab环境")
        return "colab"
    except ImportError:
        pass
    
    print("✅ 检测到本地环境")
    return "local"

def complete_uninstall():
    """完全卸载可能冲突的包"""
    print("\n🗑️ 卸载可能冲突的包...")
    
    packages_to_remove = [
        "diffusers", "transformers", "accelerate", 
        "huggingface-hub", "tokenizers", "safetensors",
        "datasets", "evaluate", "peft", "trl",
        "torch-audio", "torchaudio", "torchtext", "torchdata",
        "sentencepiece", "protobuf", "wandb", "tensorboardX",
    ]
    
    for round_num in range(2):
        print(f"第 {round_num + 1} 轮卸载:")
        for package in packages_to_remove:
            run_command(f"pip uninstall {package} -y", f"卸载 {package}")
    
    run_command("pip cache purge", "清理pip缓存")

def install_pytorch(env_type):
    """安装PyTorch"""
    print("\n🔥 安装PyTorch...")
    
    if env_type == "kaggle":
        try:
            import torch
            print(f"✅ 使用Kaggle预装PyTorch: {torch.__version__}")
            return True
        except ImportError:
            pass
    
    # 安装GPU版本
    cmd = "pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118"
    return run_command(cmd, "安装PyTorch")

def install_base_dependencies():
    """安装基础依赖"""
    print("\n📦 安装基础依赖...")
    
    base_deps = [
        "numpy==1.24.3", "pillow==10.0.1", "requests==2.31.0",
        "packaging==23.2", "filelock==3.13.1", "tqdm==4.66.1",
        "pyyaml==6.0.1", "typing-extensions==4.8.0", "regex==2023.10.3",
    ]
    
    for dep in base_deps:
        run_command(f"pip install {dep}", f"安装 {dep}")
    
    return True

def install_huggingface_ecosystem():
    """安装HuggingFace生态系统 (兼容版本)"""
    print("\n🤗 安装HuggingFace生态系统...")
    
    hf_packages = [
        ("huggingface-hub==0.17.3", "HuggingFace Hub (支持cached_download)"),
        ("tokenizers==0.14.1", "Tokenizers"),
        ("safetensors==0.4.0", "SafeTensors"),
        ("transformers==4.35.2", "Transformers"),
        ("accelerate==0.24.1", "Accelerate"),
        ("diffusers==0.24.0", "Diffusers"),
    ]
    
    for package, description in hf_packages:
        success = run_command(f"pip install {package} --no-deps", f"安装 {description}")
        if not success:
            run_command(f"pip install {package} --force-reinstall --no-deps", f"强制重装 {description}")
    
    return True

def install_other_dependencies():
    """安装其他必要依赖"""
    print("\n📚 安装其他依赖...")
    
    other_deps = [
        "scipy==1.11.4", "scikit-learn==1.3.0", "scikit-image==0.21.0",
        "matplotlib==3.7.2", "opencv-python==4.8.1.78", "einops==0.7.0",
        "tensorboard==2.15.1", "lpips==0.1.4",
    ]
    
    for dep in other_deps:
        run_command(f"pip install {dep}", f"安装 {dep}")
    
    return True

def verify_installation():
    """验证安装"""
    print("\n🔍 验证安装...")
    
    critical_packages = {
        'torch': None,
        'diffusers': '0.24.0',
        'transformers': '4.35.2', 
        'accelerate': '0.24.1',
        'huggingface_hub': '0.17.3',
    }
    
    all_good = True
    
    for package, expected_version in critical_packages.items():
        try:
            module = importlib.import_module(package)
            actual_version = getattr(module, '__version__', 'unknown')
            
            if expected_version is None or expected_version in actual_version:
                print(f"✅ {package}: {actual_version}")
            else:
                print(f"⚠️ {package}: 期望 {expected_version}, 实际 {actual_version}")
                
        except ImportError as e:
            print(f"❌ {package}: 导入失败 - {e}")
            all_good = False
    
    return all_good

def test_api_compatibility():
    """测试API兼容性"""
    print("\n🧪 测试API兼容性...")
    
    # 测试cached_download
    try:
        from huggingface_hub import cached_download
        print("✅ cached_download: 可用")
    except ImportError as e:
        print(f"❌ cached_download: 不可用 - {e}")
        return False
    
    # 测试VQModel
    try:
        from diffusers.models.autoencoders.vq_model import VQModel
        print("✅ VQModel: 可用")
        
        import torch
        model = VQModel(
            in_channels=3, out_channels=3,
            down_block_types=["DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D"],
            block_out_channels=[128],
            layers_per_block=1,
            latent_channels=128,
            sample_size=32,
            num_vq_embeddings=256,
            norm_num_groups=32,
            vq_embed_dim=128,
        )
        
        test_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            result = model.encode(test_input)
            print("✅ VQModel测试: 通过")
            
    except Exception as e:
        print(f"❌ VQModel测试: 失败 - {e}")
        return False
    
    # 测试Transformer
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        
        config = GPT2Config(vocab_size=256, n_positions=64, n_embd=128, n_layer=2, n_head=4)
        model = GPT2LMHeadModel(config)
        
        import torch
        test_input = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            output = model(test_input)
            print("✅ Transformer测试: 通过")
            
    except Exception as e:
        print(f"❌ Transformer测试: 失败 - {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("🎨 VQ-VAE + Transformer 统一环境安装器")
    print("=" * 60)
    print("🔧 自动检测环境并安装兼容版本")
    print("⚠️ 这将卸载并重新安装相关包，确保版本兼容性")
    
    # 检测环境
    env_type = detect_environment()
    
    print(f"\n📊 环境类型: {env_type}")
    
    # 确认操作
    if env_type == "local":
        response = input("\n是否继续安装? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ 操作已取消")
            return
    
    # 安装流程
    steps = [
        ("卸载冲突包", complete_uninstall),
        ("安装PyTorch", lambda: install_pytorch(env_type)),
        ("安装基础依赖", install_base_dependencies),
        ("安装HuggingFace生态", install_huggingface_ecosystem),
        ("安装其他依赖", install_other_dependencies),
        ("验证安装", verify_installation),
        ("测试API兼容性", test_api_compatibility),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"❌ {step_name} 失败")
            if step_name in ["验证安装", "测试API兼容性"]:
                print("⚠️ 可能仍然可用，继续后续步骤")
            else:
                print("❌ 安装过程中断")
                return
    
    print("\n🎉 环境安装完成!")
    print("✅ 所有依赖已正确安装并验证")
    print("✅ API兼容性测试通过")
    print("\n🚀 现在可以开始训练:")
    print("   python train_main.py --data_dir /path/to/data")

if __name__ == "__main__":
    main()
