#!/usr/bin/env python3
"""
安全的依赖测试脚本 - 避免导入冲突
专门用于验证修复结果
"""

import sys
import importlib

def test_core_imports():
    """测试核心导入，避免冲突"""
    print("🧪 安全依赖测试")
    print("=" * 40)
    
    success_count = 0
    total_tests = 0
    
    # 测试基础包
    test_cases = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('scipy', 'SciPy'),
        ('tqdm', 'TQDM'),
        ('einops', 'Einops')
    ]
    
    print("\n📦 基础包测试:")
    for module_name, display_name in test_cases:
        total_tests += 1
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {display_name}: 导入失败 - {e}")
        except Exception as e:
            print(f"⚠️  {display_name}: 导入异常 - {e}")
    
    # 测试关键包
    critical_packages = [
        ('huggingface_hub', 'HuggingFace Hub'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
        ('diffusers', 'Diffusers')
    ]
    
    print("\n🔑 关键包测试:")
    for module_name, display_name in critical_packages:
        total_tests += 1
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {display_name}: 导入失败 - {e}")
        except Exception as e:
            print(f"⚠️  {display_name}: 导入异常 - {e}")
    
    return success_count, total_tests

def test_numpy_compatibility():
    """测试NumPy兼容性"""
    print("\n🔍 NumPy兼容性测试:")
    
    try:
        import numpy as np
        print(f"✅ NumPy版本: {np.__version__}")
        
        # 检查dtypes属性
        if hasattr(np, 'dtypes'):
            print("✅ NumPy dtypes属性存在")
        else:
            print("⚠️  NumPy dtypes属性不存在 (可能导致JAX冲突)")
        
        # 基本功能测试
        test_array = np.random.randn(3, 3)
        result = np.mean(test_array)
        print(f"✅ NumPy基本功能正常: mean={result:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ NumPy测试失败: {e}")
        return False

def test_torch_functionality():
    """测试PyTorch功能"""
    print("\n🔥 PyTorch功能测试:")
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("⚠️  CUDA不可用，将使用CPU")
        
        # 基本张量操作
        test_tensor = torch.randn(2, 3)
        result = torch.mean(test_tensor)
        print(f"✅ PyTorch基本功能正常: mean={result:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        return False

def test_huggingface_hub():
    """测试HuggingFace Hub功能"""
    print("\n🤗 HuggingFace Hub测试:")
    
    try:
        import huggingface_hub
        print(f"✅ HuggingFace Hub版本: {huggingface_hub.__version__}")
        
        # 检查关键函数
        critical_functions = [
            'split_torch_state_dict_into_shards',
            'cached_download',
            'hf_hub_download'
        ]
        
        for func_name in critical_functions:
            if hasattr(huggingface_hub, func_name):
                print(f"✅ {func_name} 函数存在")
            else:
                print(f"⚠️  {func_name} 函数不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ HuggingFace Hub测试失败: {e}")
        return False

def test_diffusers_basic():
    """测试Diffusers基本功能"""
    print("\n🎨 Diffusers功能测试:")
    
    try:
        import diffusers
        print(f"✅ Diffusers版本: {diffusers.__version__}")
        
        # 测试关键类导入
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        print("✅ 关键类导入成功")
        
        # 创建小模型测试
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            sample_size=32,
        )
        print("✅ VAE模型创建成功")
        
        # 测试前向传播
        import torch
        with torch.no_grad():
            test_input = torch.randn(1, 3, 32, 32)
            latents = vae.encode(test_input).latent_dist.sample()
            reconstructed = vae.decode(latents).sample
            
            print(f"✅ VAE前向传播成功")
            print(f"   输入: {test_input.shape}")
            print(f"   潜在: {latents.shape}")
            print(f"   重建: {reconstructed.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Diffusers测试失败: {e}")
        return False

def test_training_dependencies():
    """测试训练相关依赖"""
    print("\n🏋️ 训练依赖测试:")
    
    training_modules = [
        ('einops', 'Einops'),
        ('tqdm', 'TQDM'),
        ('matplotlib', 'Matplotlib'),
        ('PIL', 'Pillow')
    ]
    
    success = 0
    for module_name, display_name in training_modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            success += 1
        except Exception as e:
            print(f"❌ {display_name}: {e}")
    
    return success == len(training_modules)

def main():
    """主测试函数"""
    print("🛡️  安全依赖测试工具")
    print("避免导入冲突，验证修复结果")
    print("=" * 50)
    
    # 1. 核心导入测试
    success_count, total_tests = test_core_imports()
    
    # 2. NumPy兼容性
    numpy_ok = test_numpy_compatibility()
    
    # 3. PyTorch功能
    torch_ok = test_torch_functionality()
    
    # 4. HuggingFace Hub
    hub_ok = test_huggingface_hub()
    
    # 5. Diffusers功能
    diffusers_ok = test_diffusers_basic()
    
    # 6. 训练依赖
    training_ok = test_training_dependencies()
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试总结:")
    print(f"   基础包导入: {success_count}/{total_tests}")
    print(f"   NumPy兼容性: {'✅' if numpy_ok else '❌'}")
    print(f"   PyTorch功能: {'✅' if torch_ok else '❌'}")
    print(f"   HuggingFace Hub: {'✅' if hub_ok else '❌'}")
    print(f"   Diffusers功能: {'✅' if diffusers_ok else '❌'}")
    print(f"   训练依赖: {'✅' if training_ok else '❌'}")
    
    # 判断整体状态
    critical_tests = [torch_ok, diffusers_ok]
    all_critical_passed = all(critical_tests)
    
    if all_critical_passed:
        print("\n🎉 核心依赖测试通过！")
        print("✅ 可以开始训练")
        print("\n📋 下一步:")
        print("   python train_kaggle.py --stage all")
    else:
        print("\n⚠️  部分关键测试失败")
        print("\n🔧 建议修复:")
        if not torch_ok:
            print("   - 运行: python quick_fix_kaggle.py")
        if not diffusers_ok:
            print("   - 运行: python fix_huggingface_hub.py")
        print("   - 或运行: python fix_versions_kaggle.py")
    
    return all_critical_passed

if __name__ == "__main__":
    main()
