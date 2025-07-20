#!/usr/bin/env python3
"""
VQ-VAE + Transformer 环境安装和验证脚本
解决API兼容性问题，确保所有依赖版本正确
"""

import os
import sys
import subprocess
import importlib
import pkg_resources
from pathlib import Path

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
        print(f"   错误: {e.stderr}")
        return False

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print("   需要Python 3.8+")
        return False

def uninstall_conflicting_packages():
    """卸载可能冲突的包"""
    print("\n🗑️ 卸载可能冲突的包...")
    
    # 需要卸载的包列表
    packages_to_uninstall = [
        "torch", "torchvision", "torchaudio",
        "diffusers", "transformers", "accelerate",
        "huggingface-hub", "tokenizers", "safetensors",
        "numpy", "pillow", "opencv-python",
        "matplotlib", "scikit-image", "scikit-learn",
        "scipy", "einops", "tqdm", "tensorboard",
        "lpips", "packaging"
    ]
    
    for package in packages_to_uninstall:
        cmd = f"pip uninstall {package} -y"
        run_command(cmd, f"卸载 {package}")
    
    print("✅ 冲突包卸载完成")

def install_pytorch():
    """安装PyTorch (CUDA 11.8版本)"""
    print("\n🔥 安装PyTorch...")
    
    # 检测CUDA
    cuda_available = False
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
        cuda_available = result.returncode == 0
    except:
        pass
    
    if cuda_available:
        print("✅ 检测到CUDA，安装GPU版本")
        cmd = "pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118"
    else:
        print("⚠️ 未检测到CUDA，安装CPU版本")
        cmd = "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0"
    
    return run_command(cmd, "安装PyTorch")

def install_huggingface():
    """安装HuggingFace生态系统"""
    print("\n🤗 安装HuggingFace生态系统...")
    
    # 按顺序安装，避免依赖冲突
    hf_packages = [
        "huggingface-hub==0.19.4",
        "tokenizers==0.15.0", 
        "safetensors==0.4.1",
        "transformers==4.36.2",
        "accelerate==0.25.0",
        "diffusers==0.25.1",
    ]
    
    for package in hf_packages:
        if not run_command(f"pip install {package}", f"安装 {package}"):
            return False
    
    return True

def install_other_dependencies():
    """安装其他依赖"""
    print("\n📦 安装其他依赖...")
    
    other_packages = [
        "numpy==1.24.3",
        "pillow==10.0.1", 
        "opencv-python==4.8.1.78",
        "matplotlib==3.7.2",
        "scikit-image==0.21.0",
        "scikit-learn==1.3.0",
        "scipy==1.11.4",
        "einops==0.7.0",
        "tqdm==4.66.1",
        "tensorboard==2.15.1",
        "packaging==23.2",
        "lpips==0.1.4",
    ]
    
    for package in other_packages:
        if not run_command(f"pip install {package}", f"安装 {package}"):
            return False
    
    return True

def verify_installation():
    """验证安装"""
    print("\n🔍 验证安装...")
    
    # 验证关键包
    critical_packages = {
        'torch': '2.1.0',
        'diffusers': '0.25.1', 
        'transformers': '4.36.2',
        'accelerate': '0.25.0',
    }
    
    all_good = True
    
    for package, expected_version in critical_packages.items():
        try:
            module = importlib.import_module(package)
            actual_version = getattr(module, '__version__', 'unknown')
            
            if expected_version in actual_version:
                print(f"✅ {package}: {actual_version}")
            else:
                print(f"❌ {package}: 期望 {expected_version}, 实际 {actual_version}")
                all_good = False
                
        except ImportError as e:
            print(f"❌ {package}: 导入失败 - {e}")
            all_good = False
    
    return all_good

def test_vq_vae_api():
    """测试VQ-VAE API兼容性"""
    print("\n🧪 测试VQ-VAE API兼容性...")
    
    try:
        # 测试diffusers VQModel
        from diffusers.models.autoencoders.vq_model import VQModel
        print("✅ VQModel导入成功")
        
        # 测试创建模型
        model = VQModel(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[128, 256],
            layers_per_block=2,
            act_fn="silu",
            latent_channels=256,
            sample_size=64,
            num_vq_embeddings=512,
            norm_num_groups=32,
            vq_embed_dim=256,
        )
        print("✅ VQModel创建成功")
        
        # 测试前向传播
        import torch
        test_input = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            result = model.encode(test_input)
            print(f"✅ VQModel编码成功: {result.latents.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ VQ-VAE API测试失败: {e}")
        return False

def test_transformer_api():
    """测试Transformer API兼容性"""
    print("\n🤖 测试Transformer API兼容性...")
    
    try:
        # 测试transformers GPT2
        from transformers import GPT2Config, GPT2LMHeadModel
        print("✅ GPT2导入成功")
        
        # 测试创建模型
        config = GPT2Config(
            vocab_size=1024,
            n_positions=256,
            n_embd=512,
            n_layer=4,
            n_head=8,
        )
        
        model = GPT2LMHeadModel(config)
        print("✅ GPT2模型创建成功")
        
        # 测试前向传播
        import torch
        test_input = torch.randint(0, 1024, (1, 32))
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ GPT2前向传播成功: {output.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformer API测试失败: {e}")
        return False

def create_environment_info():
    """创建环境信息文件"""
    print("\n📄 创建环境信息文件...")
    
    info_content = f"""# VQ-VAE + Transformer 环境信息
# 生成时间: {__import__('datetime').datetime.now()}

## Python版本
{sys.version}

## 已安装包版本
"""
    
    try:
        installed_packages = [str(d) for d in pkg_resources.working_set]
        installed_packages.sort()
        
        for package in installed_packages:
            info_content += f"{package}\n"
        
        with open("environment_info.txt", "w", encoding="utf-8") as f:
            f.write(info_content)
        
        print("✅ 环境信息保存到 environment_info.txt")
        return True
        
    except Exception as e:
        print(f"❌ 创建环境信息失败: {e}")
        return False

def main():
    """主函数"""
    print("🎨 VQ-VAE + Transformer 环境安装器")
    print("=" * 60)
    print("⚠️ 这将卸载并重新安装所有相关包，确保版本兼容性")
    
    # 确认操作
    response = input("\n是否继续? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ 操作已取消")
        return
    
    # 检查Python版本
    if not check_python_version():
        return
    
    # 步骤1: 卸载冲突包
    uninstall_conflicting_packages()
    
    # 步骤2: 安装PyTorch
    if not install_pytorch():
        print("❌ PyTorch安装失败，停止安装")
        return
    
    # 步骤3: 安装HuggingFace
    if not install_huggingface():
        print("❌ HuggingFace安装失败，停止安装")
        return
    
    # 步骤4: 安装其他依赖
    if not install_other_dependencies():
        print("❌ 其他依赖安装失败")
        return
    
    # 步骤5: 验证安装
    if not verify_installation():
        print("❌ 安装验证失败")
        return
    
    # 步骤6: 测试API
    vq_api_ok = test_vq_vae_api()
    transformer_api_ok = test_transformer_api()
    
    if not (vq_api_ok and transformer_api_ok):
        print("❌ API测试失败")
        return
    
    # 步骤7: 创建环境信息
    create_environment_info()
    
    print("\n🎉 环境安装完成!")
    print("✅ 所有依赖已正确安装并验证")
    print("✅ API兼容性测试通过")
    print("\n🚀 现在可以开始训练:")
    print("   python train_main.py --data_dir /path/to/data")

if __name__ == "__main__":
    main()
