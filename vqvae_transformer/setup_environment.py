#!/usr/bin/env python3
"""
VQ-VAE + Transformer 统一环境安装脚本
解决numpy/JAX兼容性和API版本问题
"""

import os
import sys
import subprocess
import importlib

def run_command(cmd, description="", allow_failure=False):
    """运行命令并处理错误"""
    print(f"🔄 {description}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        if e.stderr.strip():
            print(f"   错误: {e.stderr.strip()}")
        if allow_failure:
            print("⚠️ 此步骤允许失败，继续...")
            return True
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

def install_core_packages():
    """安装核心包 - 解决numpy/JAX兼容性问题"""
    print("\n🔧 安装核心包...")
    
    # 先安装兼容的numpy版本 - 解决JAX兼容性问题
    success = run_command("pip install 'numpy>=1.26.0,<2.0.0'", "安装兼容的numpy")
    if not success:
        run_command("pip install numpy==1.26.4", "安装numpy (指定版本)")
    
    # 安装其他核心依赖
    core_deps = [
        "pillow>=9.0.0",
        "requests>=2.28.0", 
        "packaging>=21.0",
        "filelock>=3.0.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "typing-extensions>=4.0.0",
        "regex>=2022.0.0",
    ]
    
    for dep in core_deps:
        run_command(f"pip install '{dep}'", f"安装 {dep.split('>=')[0]}", allow_failure=True)
    
    return True

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
    
    # 安装PyTorch
    cmd = "pip install 'torch>=2.0.0' 'torchvision>=0.15.0' 'torchaudio>=2.0.0' --index-url https://download.pytorch.org/whl/cu118"
    success = run_command(cmd, "安装PyTorch (CUDA)")
    
    if not success:
        cmd = "pip install 'torch>=2.0.0' 'torchvision>=0.15.0' 'torchaudio>=2.0.0'"
        run_command(cmd, "安装PyTorch (CPU)")
    
    return True

def install_huggingface_stack():
    """安装HuggingFace技术栈"""
    print("\n🤗 安装HuggingFace技术栈...")
    
    # 按依赖顺序安装
    hf_packages = [
        ("huggingface-hub>=0.17.0,<0.25.0", "HuggingFace Hub"),
        ("tokenizers>=0.14.0,<0.20.0", "Tokenizers"),
        ("safetensors>=0.4.0,<0.5.0", "SafeTensors"),
        ("transformers>=4.35.0,<4.45.0", "Transformers"),
        ("accelerate>=0.24.0,<0.35.0", "Accelerate"),
        ("diffusers>=0.24.0,<0.30.0", "Diffusers"),
    ]
    
    for package_spec, name in hf_packages:
        success = run_command(f"pip install '{package_spec}'", f"安装 {name}")
        if not success:
            package_name = package_spec.split('>=')[0].split('<')[0]
            run_command(f"pip install {package_name}", f"安装 {name} (无版本限制)", allow_failure=True)
    
    return True

def install_additional_deps():
    """安装其他必要依赖"""
    print("\n📚 安装其他依赖...")
    
    additional_deps = [
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0", 
        "scikit-image>=0.19.0",
        "matplotlib>=3.5.0",
        "opencv-python>=4.6.0",
        "einops>=0.6.0",
        "tensorboard>=2.10.0",
        "lpips>=0.1.4",
    ]
    
    for dep in additional_deps:
        run_command(f"pip install '{dep}'", f"安装 {dep.split('>=')[0]}", allow_failure=True)
    
    return True

def test_critical_imports():
    """测试关键导入"""
    print("\n🧪 测试关键导入...")
    
    critical_tests = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("huggingface_hub", "HuggingFace Hub"),
        ("accelerate", "Accelerate"),
    ]
    
    all_good = True
    
    for module_name, display_name in critical_tests:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
        except ImportError as e:
            print(f"❌ {display_name}: 导入失败 - {e}")
            all_good = False
    
    return all_good

def test_vqmodel_api():
    """测试VQModel API"""
    print("\n🎨 测试VQModel API...")
    
    # 尝试不同的导入路径
    VQModel = None
    
    import_attempts = [
        ("diffusers.models.autoencoders.vq_model", "新版API"),
        ("diffusers.models.vq_model", "旧版API"),
        ("diffusers", "直接导入"),
    ]
    
    for module_path, description in import_attempts:
        try:
            if module_path == "diffusers":
                from diffusers import VQModel
            else:
                module = importlib.import_module(module_path)
                VQModel = getattr(module, 'VQModel')
            
            print(f"✅ VQModel导入成功: {description}")
            break
        except (ImportError, AttributeError):
            continue
    
    if VQModel is None:
        print("❌ VQModel: 所有导入路径都失败")
        return False
    
    # 测试创建和使用
    try:
        import torch
        model = VQModel(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D"],
            block_out_channels=[64],
            layers_per_block=1,
            latent_channels=64,
            sample_size=32,
            num_vq_embeddings=128,
            norm_num_groups=32,
            vq_embed_dim=64,
        )
        
        test_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            result = model.encode(test_input)
            decoded = model.decode(result.latents)
            print(f"✅ VQModel测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ VQModel创建/测试失败: {e}")
        return False

def test_transformer_api():
    """测试Transformer API"""
    print("\n🤖 测试Transformer API...")
    
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        
        config = GPT2Config(
            vocab_size=256,
            n_positions=64,
            n_embd=128,
            n_layer=2,
            n_head=4,
            use_cache=False,
        )
        
        model = GPT2LMHeadModel(config)
        
        import torch
        test_input = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ Transformer测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformer测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🎨 VQ-VAE + Transformer 统一环境安装器")
    print("=" * 60)
    print("🔧 解决numpy/JAX兼容性和API版本问题")
    
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
        ("安装核心包", install_core_packages),
        ("安装PyTorch", lambda: install_pytorch(env_type)),
        ("安装HuggingFace技术栈", install_huggingface_stack),
        ("安装其他依赖", install_additional_deps),
        ("测试关键导入", test_critical_imports),
        ("测试VQModel API", test_vqmodel_api),
        ("测试Transformer API", test_transformer_api),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        success = step_func()
        
        if not success:
            print(f"❌ {step_name} 失败")
            failed_steps.append(step_name)
        else:
            print(f"✅ {step_name} 成功")
    
    # 总结
    print(f"\n{'='*20} 安装总结 {'='*20}")
    
    if not failed_steps:
        print("🎉 环境安装完全成功!")
        print("✅ 所有组件正常工作")
        print("\n🚀 现在可以开始训练:")
        print("   python train_main.py --data_dir /path/to/data")
    else:
        print(f"⚠️ 部分步骤失败: {', '.join(failed_steps)}")
        
        if "测试关键导入" not in failed_steps:
            print("✅ 基础环境安装成功，可以尝试运行")
            print("💡 建议重启Python内核后再次测试")
        else:
            print("❌ 基础环境有问题，建议:")
            print("1. 重启Python内核")
            print("2. 重新运行此脚本")
            print("3. 检查网络连接")

if __name__ == "__main__":
    main()
