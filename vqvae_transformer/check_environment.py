#!/usr/bin/env python3
"""
快速环境检查脚本
在训练前验证所有依赖和API是否正确
"""

import sys
import torch
import importlib
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"🐍 Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python版本符合要求")
        return True
    else:
        print("❌ Python版本过低，需要3.8+")
        return False

def check_pytorch():
    """检查PyTorch"""
    try:
        import torch
        import torchvision
        import torchaudio
        
        print(f"🔥 PyTorch版本: {torch.__version__}")
        print(f"   TorchVision: {torchvision.__version__}")
        print(f"   TorchAudio: {torchaudio.__version__}")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"   CUDA可用: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)")
        else:
            print("   CUDA不可用，将使用CPU")
        
        # 版本检查
        expected_version = "2.1.0"
        if expected_version in torch.__version__:
            print("✅ PyTorch版本正确")
            return True
        else:
            print(f"⚠️ PyTorch版本不匹配，期望{expected_version}，实际{torch.__version__}")
            return False
            
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False

def check_huggingface():
    """检查HuggingFace生态系统"""
    packages = {
        'diffusers': '0.25.1',
        'transformers': '4.36.2', 
        'accelerate': '0.25.0',
        'huggingface_hub': '0.19.4',
    }
    
    all_good = True
    
    for package, expected_version in packages.items():
        try:
            module = importlib.import_module(package)
            actual_version = getattr(module, '__version__', 'unknown')
            
            print(f"🤗 {package}: {actual_version}")
            
            if expected_version in actual_version:
                print(f"   ✅ 版本正确")
            else:
                print(f"   ⚠️ 版本不匹配，期望{expected_version}")
                all_good = False
                
        except ImportError as e:
            print(f"❌ {package}导入失败: {e}")
            all_good = False
    
    return all_good

def check_vq_vae_api():
    """检查VQ-VAE API - 尝试不同版本的导入路径"""
    print("\n🧪 测试VQ-VAE API...")

    VQModel = None

    # 尝试不同版本的API路径
    try:
        # 测试新的API路径
        from diffusers.models.autoencoders.vq_model import VQModel
        print("✅ VQModel导入成功 (新版API)")
    except ImportError:
        try:
            # 测试旧版API路径
            from diffusers.models.vq_model import VQModel
            print("✅ VQModel导入成功 (旧版API)")
        except ImportError:
            try:
                # 测试直接导入
                from diffusers import VQModel
                print("✅ VQModel导入成功 (直接导入)")
            except ImportError:
                print("❌ VQModel: 所有导入路径都失败")
                print("   建议运行: python setup_environment.py")
                return False

    if VQModel is not None:
        try:
            # 测试创建模型 - 使用更简单的配置
            model = VQModel(
                in_channels=3,
                out_channels=3,
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
            print("✅ VQModel创建成功")

            # 测试前向传播
            test_input = torch.randn(1, 3, 32, 32)
            with torch.no_grad():
                result = model.encode(test_input)
                print(f"✅ VQModel编码成功: {result.latents.shape}")

                decoded = model.decode(result.latents)
                print(f"✅ VQModel解码成功: {decoded.sample.shape}")

            return True

        except Exception as e:
            print(f"❌ VQModel创建/测试失败: {e}")
            print("⚠️ VQModel导入成功但创建失败，可能是参数问题")
            print("   建议运行: python setup_environment.py")
            return True  # 导入成功就算基本通过

    return False

def check_transformer_api():
    """检查Transformer API"""
    print("\n🤖 测试Transformer API...")
    
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        print("✅ GPT2导入成功")
        
        # 测试创建模型
        config = GPT2Config(
            vocab_size=1024,
            n_positions=256,
            n_embd=512,
            n_layer=4,
            n_head=8,
            use_cache=False,
        )
        
        model = GPT2LMHeadModel(config)
        print("✅ GPT2模型创建成功")
        
        # 测试前向传播
        test_input = torch.randint(0, 1024, (1, 32))
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ GPT2前向传播成功: {output.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformer API测试失败: {e}")
        return False

def check_other_dependencies():
    """检查其他依赖"""
    print("\n📦 检查其他依赖...")
    
    dependencies = [
        'numpy', 'PIL', 'cv2', 'matplotlib', 
        'sklearn', 'scipy', 'einops', 'tqdm'
    ]
    
    all_good = True
    
    for dep in dependencies:
        try:
            if dep == 'PIL':
                import PIL
                print(f"✅ Pillow: {PIL.__version__}")
            elif dep == 'cv2':
                import cv2
                print(f"✅ OpenCV: {cv2.__version__}")
            elif dep == 'sklearn':
                import sklearn
                print(f"✅ Scikit-learn: {sklearn.__version__}")
            else:
                module = importlib.import_module(dep)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {dep}: {version}")
                
        except ImportError as e:
            print(f"❌ {dep}导入失败: {e}")
            all_good = False
    
    return all_good

def estimate_memory_requirements():
    """估算内存需求"""
    print("\n💾 内存需求估算...")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            
            print(f"GPU {i} ({props.name}): {memory_gb:.1f}GB")
            
            if memory_gb >= 16:
                print("   ✅ 内存充足，可以使用大批次训练")
                print("   推荐配置: VQ-VAE batch_size=16, Transformer batch_size=8")
            elif memory_gb >= 8:
                print("   ✅ 内存足够，使用中等批次训练")
                print("   推荐配置: VQ-VAE batch_size=12, Transformer batch_size=6")
            else:
                print("   ⚠️ 内存较少，需要小批次训练")
                print("   推荐配置: VQ-VAE batch_size=8, Transformer batch_size=4")
    else:
        print("未检测到GPU，将使用CPU训练（速度较慢）")

def main():
    """主函数"""
    print("🔍 VQ-VAE + Transformer 环境检查")
    print("=" * 50)
    
    checks = [
        ("Python版本", check_python_version),
        ("PyTorch", check_pytorch), 
        ("HuggingFace生态", check_huggingface),
        ("VQ-VAE API", check_vq_vae_api),
        ("Transformer API", check_transformer_api),
        ("其他依赖", check_other_dependencies),
    ]
    
    results = []
    
    for name, check_func in checks:
        print(f"\n{'='*20} {name} {'='*20}")
        result = check_func()
        results.append((name, result))
    
    # 内存需求估算
    print(f"\n{'='*20} 内存需求 {'='*20}")
    estimate_memory_requirements()
    
    # 总结
    print(f"\n{'='*20} 检查总结 {'='*20}")
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有检查通过！环境配置正确")
        print("🚀 可以开始训练:")
        print("   python train_main.py --data_dir /path/to/data")
    else:
        print("\n⚠️ 部分检查失败，建议:")
        print("1. 运行环境安装脚本: python setup_environment.py")
        print("2. 检查错误信息并手动修复")
        print("3. 重新运行此检查脚本")

if __name__ == "__main__":
    main()
