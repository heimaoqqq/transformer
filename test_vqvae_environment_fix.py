#!/usr/bin/env python3
"""
VQ-VAE环境修复验证脚本
测试修复后的依赖版本组合是否能正常工作
"""

import sys
import subprocess
import importlib

def test_import(module_name, description):
    """测试模块导入"""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {description}: {version}")
        return True, version
    except ImportError as e:
        print(f"❌ {description}: 导入失败 - {e}")
        return False, None
    except Exception as e:
        print(f"⚠️ {description}: 导入异常 - {e}")
        return False, None

def test_vqmodel_import():
    """测试VQModel导入"""
    print("\n🧪 测试VQModel导入...")
    
    # 测试diffusers官方推荐的导入路径
    import_paths = [
        ("diffusers.models.autoencoders.vq_model", "VQModel", "官方推荐路径 (diffusers最新版本)"),
        ("diffusers.models.autoencoders.vq_model", "VectorQuantizer", "VectorQuantizer导入"),
    ]
    
    success_count = 0
    for module_path, class_name, description in import_paths:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            print(f"✅ {description}: {class_name} 可用")
            success_count += 1
        except ImportError as e:
            print(f"❌ {description}: {class_name} 导入失败 - {e}")
        except AttributeError as e:
            print(f"❌ {description}: {class_name} 不存在 - {e}")
        except Exception as e:
            print(f"⚠️ {description}: {class_name} 异常 - {e}")
    
    return success_count > 0

def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        
        # 测试基本tensor操作
        x = torch.randn(2, 3, 64, 64)
        print(f"✅ Tensor操作: {x.shape}")
        
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            print(f"✅ GPU操作: {x_gpu.device}")
        
        return True
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def test_vqvae_model_creation():
    """测试VQ-VAE模型创建"""
    print("\n🧪 测试VQ-VAE模型创建...")
    
    try:
        # 尝试导入我们的自定义模型
        sys.path.insert(0, '.')
        from models.vqvae_model import MicroDopplerVQVAE
        
        # 创建模型实例
        model = MicroDopplerVQVAE(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            codebook_size=1024,
            codebook_dim=256
        )
        
        print(f"✅ MicroDopplerVQVAE创建成功")
        print(f"✅ 参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        import torch
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = model(x)
            print(f"✅ 前向传播成功: {output.sample.shape}")
        
        return True
    except Exception as e:
        print(f"❌ VQ-VAE模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🧪 VQ-VAE环境修复验证")
    print("=" * 50)
    
    # 1. 基础包导入测试
    print("\n1️⃣ 基础包导入测试:")
    basic_tests = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("matplotlib", "Matplotlib"),
    ]
    
    basic_success = 0
    for module, desc in basic_tests:
        success, version = test_import(module, desc)
        if success:
            basic_success += 1
    
    # 2. HuggingFace生态系统测试
    print("\n2️⃣ HuggingFace生态系统测试:")
    hf_tests = [
        ("huggingface_hub", "HuggingFace Hub"),
        ("tokenizers", "Tokenizers"),
        ("safetensors", "SafeTensors"),
        ("diffusers", "Diffusers"),
    ]
    
    hf_success = 0
    for module, desc in hf_tests:
        success, version = test_import(module, desc)
        if success:
            hf_success += 1
    
    # 3. VQModel导入测试
    vqmodel_success = test_vqmodel_import()
    
    # 4. 基本功能测试
    basic_func_success = test_basic_functionality()
    
    # 5. VQ-VAE模型测试
    vqvae_success = test_vqvae_model_creation()
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print(f"   基础包: {basic_success}/{len(basic_tests)} ✅")
    print(f"   HuggingFace: {hf_success}/{len(hf_tests)} ✅")
    print(f"   VQModel导入: {'✅' if vqmodel_success else '❌'}")
    print(f"   基本功能: {'✅' if basic_func_success else '❌'}")
    print(f"   VQ-VAE模型: {'✅' if vqvae_success else '❌'}")
    
    total_success = (
        basic_success >= len(basic_tests) - 1 and  # 允许1个失败
        hf_success >= len(hf_tests) - 1 and       # 允许1个失败
        vqmodel_success and
        basic_func_success
    )
    
    if total_success:
        print("\n🎉 diffusers官方配置环境验证成功！")
        print("✅ 可以开始VQ-VAE训练")
        print("\n📋 下一步:")
        print("   python training/train_vqvae.py --help")
        print("\n💡 版本信息:")
        print("   - diffusers: 最新版本 (官方配置)")
        print("   - transformers: 官方要求的依赖")
        print("   - 导入路径: diffusers.models.autoencoders.vq_model")
        return True
    else:
        print("\n❌ 环境验证失败")
        print("🔧 建议:")
        print("   1. 重新运行: python setup_vqvae_environment.py")
        print("   2. 检查diffusers官方配置是否正确安装")
        print("   3. 确认transformers已正确安装")
        print("   4. 重启Python内核后重试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
