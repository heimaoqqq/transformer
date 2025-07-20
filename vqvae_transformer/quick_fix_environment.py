#!/usr/bin/env python3
"""
快速环境修复脚本
专门解决diffusers兼容性问题，跳过卸载步骤
"""

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

def quick_install_huggingface():
    """快速安装HuggingFace兼容版本"""
    print("🚀 快速安装HuggingFace兼容版本...")
    
    # 直接安装兼容版本，覆盖现有版本
    hf_packages = [
        "huggingface-hub==0.17.3",
        "tokenizers==0.14.1", 
        "safetensors==0.4.0",
        "transformers==4.35.2",
        "accelerate==0.24.1",
        "diffusers==0.24.0",
    ]
    
    for package in hf_packages:
        # 使用--force-reinstall确保覆盖
        success = run_command(f"pip install {package} --force-reinstall", f"强制安装 {package}")
        if not success:
            print(f"⚠️ {package} 安装失败，继续...")
    
    return True

def quick_install_essentials():
    """快速安装必要依赖"""
    print("📦 快速安装必要依赖...")
    
    essentials = [
        "numpy", "pillow", "requests", "tqdm", 
        "einops", "scipy", "matplotlib"
    ]
    
    for package in essentials:
        run_command(f"pip install {package}", f"安装 {package}")
    
    return True

def test_critical_imports():
    """测试关键导入"""
    print("\n🧪 测试关键导入...")
    
    tests = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("huggingface_hub", "HuggingFace Hub"),
    ]
    
    all_good = True
    
    for module_name, display_name in tests:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
        except ImportError as e:
            print(f"❌ {display_name}: 导入失败 - {e}")
            all_good = False
    
    return all_good

def test_cached_download():
    """测试cached_download函数"""
    print("\n🔍 测试cached_download...")
    
    try:
        from huggingface_hub import cached_download
        print("✅ cached_download: 可用")
        return True
    except ImportError as e:
        print(f"❌ cached_download: 不可用 - {e}")
        return False

def test_vqmodel():
    """测试VQModel"""
    print("\n🎨 测试VQModel...")
    
    try:
        from diffusers.models.autoencoders.vq_model import VQModel
        print("✅ VQModel: 导入成功")
        
        # 简单测试
        import torch
        model = VQModel(
            in_channels=3, out_channels=3,
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
            print("✅ VQModel: 测试通过")
            return True
            
    except Exception as e:
        print(f"❌ VQModel: 测试失败 - {e}")
        return False

def main():
    """主函数"""
    print("🔧 快速环境修复脚本")
    print("=" * 50)
    print("🎯 专门解决diffusers兼容性问题")
    print("⚡ 跳过卸载步骤，直接覆盖安装")
    
    # 步骤1: 快速安装HuggingFace
    print(f"\n{'='*20} 安装HuggingFace生态 {'='*20}")
    quick_install_huggingface()
    
    # 步骤2: 安装必要依赖
    print(f"\n{'='*20} 安装必要依赖 {'='*20}")
    quick_install_essentials()
    
    # 步骤3: 测试导入
    print(f"\n{'='*20} 测试导入 {'='*20}")
    imports_ok = test_critical_imports()
    
    # 步骤4: 测试cached_download
    print(f"\n{'='*20} 测试cached_download {'='*20}")
    cached_download_ok = test_cached_download()
    
    # 步骤5: 测试VQModel
    print(f"\n{'='*20} 测试VQModel {'='*20}")
    vqmodel_ok = test_vqmodel()
    
    # 总结
    print(f"\n{'='*20} 修复总结 {'='*20}")
    
    if imports_ok and cached_download_ok and vqmodel_ok:
        print("🎉 环境修复成功!")
        print("✅ 所有关键组件正常工作")
        print("\n🚀 现在可以开始训练:")
        print("   python train_main.py --data_dir /path/to/data")
    else:
        print("⚠️ 部分组件仍有问题")
        
        if not imports_ok:
            print("❌ 基础导入失败")
        if not cached_download_ok:
            print("❌ cached_download不可用")
        if not vqmodel_ok:
            print("❌ VQModel测试失败")
        
        print("\n💡 建议:")
        print("1. 重启Python内核/环境")
        print("2. 运行完整安装脚本: python setup_environment.py")
        print("3. 检查Python版本是否为3.8+")
        
        # 显示当前版本信息
        print(f"\n📊 当前版本信息:")
        for module_name in ["diffusers", "transformers", "huggingface_hub"]:
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"   {module_name}: {version}")
            except ImportError:
                print(f"   {module_name}: 未安装")

if __name__ == "__main__":
    main()
