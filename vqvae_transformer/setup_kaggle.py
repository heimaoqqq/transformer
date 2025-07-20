#!/usr/bin/env python3
"""
Kaggle专用环境安装脚本
针对Kaggle环境优化，确保API兼容性
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
        if result.stdout:
            print(f"   输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        print(f"   错误: {e.stderr}")
        return False

def check_kaggle_environment():
    """检查Kaggle环境"""
    print("🏠 检查Kaggle环境...")
    
    # 检查是否在Kaggle环境中
    kaggle_indicators = [
        os.path.exists('/kaggle'),
        'KAGGLE_KERNEL_RUN_TYPE' in os.environ,
        'KAGGLE_URL_BASE' in os.environ,
    ]
    
    if any(kaggle_indicators):
        print("✅ 检测到Kaggle环境")
        
        # 检查GPU
        if os.path.exists('/opt/bin/nvidia-smi'):
            result = subprocess.run('nvidia-smi', shell=True, capture_output=True)
            if result.returncode == 0:
                print("✅ GPU可用")
                return "gpu"
            else:
                print("⚠️ GPU不可用")
                return "cpu"
        else:
            print("⚠️ 未检测到nvidia-smi")
            return "cpu"
    else:
        print("⚠️ 不在Kaggle环境中")
        return "local"

def uninstall_kaggle_conflicts():
    """卸载Kaggle环境中可能冲突的包"""
    print("\n🗑️ 卸载Kaggle环境中的冲突包...")
    
    # Kaggle预装的包可能版本不兼容
    packages_to_uninstall = [
        "diffusers", "transformers", "accelerate",
        "huggingface-hub", "tokenizers", "safetensors"
    ]
    
    for package in packages_to_uninstall:
        cmd = f"pip uninstall {package} -y"
        run_command(cmd, f"卸载 {package}")

def install_kaggle_pytorch():
    """在Kaggle环境中安装PyTorch"""
    print("\n🔥 检查PyTorch...")
    
    try:
        import torch
        print(f"✅ PyTorch已安装: {torch.__version__}")
        
        # 检查版本是否兼容
        if "2.1" in torch.__version__ or "2.0" in torch.__version__:
            print("✅ PyTorch版本兼容")
            return True
        else:
            print(f"⚠️ PyTorch版本可能不兼容: {torch.__version__}")
            
    except ImportError:
        print("❌ PyTorch未安装")
        
    # 在Kaggle中通常不需要重装PyTorch，因为预装版本通常可用
    print("ℹ️ 使用Kaggle预装的PyTorch版本")
    return True

def install_kaggle_huggingface():
    """在Kaggle环境中安装HuggingFace"""
    print("\n🤗 安装HuggingFace生态系统...")
    
    # 使用--no-deps避免依赖冲突，然后手动安装必要依赖
    hf_packages = [
        "huggingface-hub==0.19.4",
        "tokenizers==0.15.0",
        "safetensors==0.4.1", 
        "transformers==4.36.2",
        "accelerate==0.25.0",
        "diffusers==0.25.1",
    ]
    
    for package in hf_packages:
        # 先尝试正常安装
        if not run_command(f"pip install {package}", f"安装 {package}"):
            # 如果失败，尝试强制安装
            run_command(f"pip install {package} --force-reinstall --no-deps", f"强制安装 {package}")
    
    return True

def install_kaggle_other():
    """安装其他必要依赖"""
    print("\n📦 安装其他依赖...")
    
    # 只安装可能缺失的包
    other_packages = [
        "einops==0.7.0",
        "lpips==0.1.4",
    ]
    
    for package in other_packages:
        run_command(f"pip install {package}", f"安装 {package}")
    
    return True

def verify_kaggle_installation():
    """验证Kaggle环境中的安装"""
    print("\n🔍 验证安装...")
    
    critical_packages = {
        'torch': None,  # 不检查具体版本，使用Kaggle预装版本
        'diffusers': '0.25.1',
        'transformers': '4.36.2', 
        'accelerate': '0.25.0',
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
                # 在Kaggle环境中，版本不完全匹配可能仍然可用
                
        except ImportError as e:
            print(f"❌ {package}: 导入失败 - {e}")
            all_good = False
    
    return all_good

def test_kaggle_apis():
    """测试Kaggle环境中的API"""
    print("\n🧪 测试API兼容性...")
    
    # 测试VQ-VAE API
    try:
        from diffusers.models.autoencoders.vq_model import VQModel
        print("✅ VQModel API可用")
        
        # 简单测试
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
            print("✅ VQModel测试通过")
            
    except Exception as e:
        print(f"❌ VQModel测试失败: {e}")
        return False
    
    # 测试Transformer API
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        
        config = GPT2Config(vocab_size=256, n_positions=64, n_embd=128, n_layer=2, n_head=4)
        model = GPT2LMHeadModel(config)
        
        test_input = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            output = model(test_input)
            print("✅ Transformer测试通过")
            
    except Exception as e:
        print(f"❌ Transformer测试失败: {e}")
        return False
    
    return True

def optimize_kaggle_settings():
    """优化Kaggle环境设置"""
    print("\n⚙️ 优化Kaggle设置...")
    
    # 设置环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # 检查可用内存
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU内存: {gpu_memory:.1f}GB")
            
            if gpu_memory >= 15:
                print("   推荐配置: VQ-VAE batch_size=16, Transformer batch_size=8")
            else:
                print("   推荐配置: VQ-VAE batch_size=12, Transformer batch_size=6")
        else:
            print("⚠️ 未检测到GPU")
            
    except Exception as e:
        print(f"⚠️ 内存检查失败: {e}")

def main():
    """主函数"""
    print("🏠 Kaggle VQ-VAE + Transformer 环境配置")
    print("=" * 60)
    
    # 检查环境
    env_type = check_kaggle_environment()
    
    if env_type == "local":
        print("⚠️ 不在Kaggle环境中，建议使用 setup_environment.py")
        return
    
    # 卸载冲突包
    uninstall_kaggle_conflicts()
    
    # 检查PyTorch
    if not install_kaggle_pytorch():
        print("❌ PyTorch配置失败")
        return
    
    # 安装HuggingFace
    if not install_kaggle_huggingface():
        print("❌ HuggingFace安装失败")
        return
    
    # 安装其他依赖
    install_kaggle_other()
    
    # 验证安装
    if not verify_kaggle_installation():
        print("⚠️ 部分包验证失败，但可能仍然可用")
    
    # 测试API
    if not test_kaggle_apis():
        print("❌ API测试失败")
        return
    
    # 优化设置
    optimize_kaggle_settings()
    
    print("\n🎉 Kaggle环境配置完成!")
    print("✅ 可以开始训练:")
    print("   python train_main.py --data_dir /kaggle/input/dataset")
    
    # 保存配置信息
    with open("/kaggle/working/kaggle_setup_complete.txt", "w") as f:
        f.write("Kaggle VQ-VAE + Transformer environment setup completed successfully\n")
        f.write(f"Python: {sys.version}\n")
        
        try:
            import torch, diffusers, transformers
            f.write(f"PyTorch: {torch.__version__}\n")
            f.write(f"Diffusers: {diffusers.__version__}\n") 
            f.write(f"Transformers: {transformers.__version__}\n")
        except:
            pass
    
    print("📄 配置信息保存到: /kaggle/working/kaggle_setup_complete.txt")

if __name__ == "__main__":
    main()
