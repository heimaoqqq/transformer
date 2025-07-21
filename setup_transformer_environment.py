#!/usr/bin/env python3
"""
Transformer阶段专用环境配置脚本
专注于transformers和序列生成依赖，使用最新版本获得最佳性能

功能：
- 安装PyTorch GPU版本
- 安装最新版transformers和huggingface_hub
- 安装序列生成和训练加速依赖
- 检查VQ-VAE模型可用性
- 测试Transformer环境完整性
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

def install_transformer_dependencies():
    """安装Transformer专用依赖"""
    print("🤖 安装Transformer专用依赖...")
    
    # 清理可能冲突的包
    run_command("pip uninstall -y transformers accelerate huggingface_hub", "清理可能冲突的包")
    run_command("pip cache purge", "清理pip缓存")
    
    # Transformer专用包 (使用最新版本)
    transformer_packages = [
        ("huggingface_hub>=0.30.0", "HuggingFace Hub (最新版本)"),
        ("transformers>=4.50.0", "Transformers (最新版本)"),
        ("accelerate>=0.25.0", "Accelerate (训练加速)"),
        ("safetensors>=0.3.1", "SafeTensors"),
        ("tokenizers>=0.15.0", "Tokenizers"),
    ]
    
    success_count = 0
    for package, description in transformer_packages:
        if run_command(f"pip install '{package}' --force-reinstall --no-cache-dir", f"安装 {description}"):
            success_count += 1
    
    print(f"\n📊 Transformer依赖安装结果: {success_count}/{len(transformer_packages)} 成功")
    return success_count >= len(transformer_packages) - 1

def install_other_dependencies():
    """安装其他依赖"""
    print("📚 安装其他依赖...")
    
    other_packages = [
        "numpy==1.26.4",
        "scipy==1.11.4", 
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "einops==0.7.0",
        "tqdm",
        "pillow",
    ]
    
    success_count = 0
    for package in other_packages:
        if run_command(f"pip install {package}", f"安装 {package}"):
            success_count += 1
    
    print(f"📊 其他依赖安装结果: {success_count}/{len(other_packages)} 成功")
    return success_count >= len(other_packages) - 2

def test_transformer_environment():
    """测试Transformer环境"""
    print("🧪 测试Transformer环境...")
    
    # 清理模块缓存
    modules_to_clear = ['torch', 'transformers', 'huggingface_hub', 'accelerate']
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    tests = [
        ("PyTorch", "torch"),
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
    
    # 测试GPT2
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        print("✅ GPT2: 可用 (Transformer核心组件)")
        
        # 测试创建模型
        config = GPT2Config(vocab_size=1024, n_positions=256, n_embd=512, n_layer=4, n_head=8)
        model = GPT2LMHeadModel(config)
        print("✅ GPT2模型: 创建成功")
        
    except Exception as e:
        print(f"❌ GPT2: 测试失败 - {e}")
    
    print(f"\n📊 Transformer环境测试结果: {success_count}/{len(tests)} 成功")
    return success_count >= len(tests) - 1

def check_vqvae_model():
    """检查VQ-VAE模型是否存在"""
    print("🔍 检查VQ-VAE模型...")
    
    possible_paths = [
        "/kaggle/working/outputs/vqvae_transformer/vqvae",
        "./outputs/vqvae_transformer/vqvae",
        "./outputs/vqvae",
    ]
    
    for path in possible_paths:
        vqvae_path = Path(path)
        if (vqvae_path / "best_model.pth").exists() or (vqvae_path / "final_model.pth").exists():
            print(f"✅ 找到VQ-VAE模型: {vqvae_path}")
            return True
    
    print("⚠️ 未找到VQ-VAE模型")
    print("   请先运行VQ-VAE训练阶段")
    return False

def main():
    """主函数"""
    print("🤖 Transformer阶段环境配置脚本")
    print("=" * 50)
    print("🎯 专用于Transformer训练的环境配置")
    print("💡 使用最新版本获得最佳性能")
    
    steps = [
        ("安装PyTorch", install_pytorch),
        ("安装Transformer依赖", install_transformer_dependencies),
        ("安装其他依赖", install_other_dependencies),
        ("测试环境", test_transformer_environment),
        ("检查VQ-VAE模型", check_vqvae_model),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            if step_name == "检查VQ-VAE模型":
                print("⚠️ VQ-VAE模型检查失败，但环境配置完成")
                break
            else:
                print(f"❌ {step_name} 失败，停止配置")
                return
    
    print("\n🎉 Transformer环境配置完成!")
    print("✅ 可以开始Transformer训练")
    print("\n🚀 使用方法:")
    print("   python train_main.py --skip_vqvae --data_dir /kaggle/input/dataset")
    print("   或者")
    print("   python training/train_transformer.py --vqvae_path ./outputs/vqvae --data_dir /kaggle/input/dataset")

if __name__ == "__main__":
    main()
