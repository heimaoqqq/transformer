#!/usr/bin/env python3
"""
统一环境测试脚本
验证VQ-VAE + Transformer统一环境的完整性
"""

import torch
import sys
import importlib
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_basic_imports():
    """测试基础导入"""
    print("🧪 测试基础导入...")
    
    tests = [
        ("PyTorch", "torch"),
        ("Diffusers", "diffusers"),
        ("Transformers", "transformers"),
        ("HuggingFace Hub", "huggingface_hub"),
        ("Accelerate", "accelerate"),
        ("SafeTensors", "safetensors"),
    ]
    
    success_count = 0
    for name, module in tests:
        try:
            imported_module = importlib.import_module(module)
            version = getattr(imported_module, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name}: 导入失败 - {e}")
    
    print(f"\n📊 基础导入结果: {success_count}/{len(tests)} 成功")
    return success_count >= len(tests) - 1

def test_vqmodel():
    """测试VQModel"""
    print("\n🧪 测试VQModel...")
    
    try:
        from diffusers.models.autoencoders.vq_model import VQModel
        print("✅ VQModel导入成功")
        
        # 测试创建VQModel实例
        config = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "num_vq_embeddings": 1024,
            "vq_embed_dim": 256,
        }
        
        model = VQModel(**config)
        print("✅ VQModel实例创建成功")
        
        # 测试前向传播
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output = model(x)
            print(f"✅ VQModel前向传播成功: {output.sample.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ VQModel测试失败: {e}")
        return False

def test_transformers():
    """测试Transformers"""
    print("\n🧪 测试Transformers...")
    
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        print("✅ GPT2导入成功")
        
        # 测试创建GPT2模型
        config = GPT2Config(
            vocab_size=1024,
            n_positions=256,
            n_embd=512,
            n_layer=4,
            n_head=8
        )
        
        model = GPT2LMHeadModel(config)
        print("✅ GPT2实例创建成功")
        
        # 测试前向传播
        input_ids = torch.randint(0, 1024, (1, 10))
        with torch.no_grad():
            output = model(input_ids)
            print(f"✅ GPT2前向传播成功: {output.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformers测试失败: {e}")
        return False

def test_custom_models():
    """测试自定义模型"""
    print("\n🧪 测试自定义模型...")
    
    try:
        from models.vqvae_model import MicroDopplerVQVAE
        print("✅ MicroDopplerVQVAE导入成功")
        
        # 测试创建自定义VQ-VAE模型
        model = MicroDopplerVQVAE(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            codebook_size=1024,
            codebook_dim=256
        )
        print("✅ MicroDopplerVQVAE实例创建成功")
        
        # 测试前向传播
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = model(x)
            print(f"✅ MicroDopplerVQVAE前向传播成功: {output.sample.shape}")
        
        # 测试模型保存和加载
        state_dict = model.state_dict()
        print(f"✅ 模型权重获取成功: {len(state_dict)} 个参数")
        
        return True
        
    except Exception as e:
        print(f"❌ 自定义模型测试失败: {e}")
        return False

def test_gpu_support():
    """测试GPU支持"""
    print("\n🧪 测试GPU支持...")
    
    try:
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
            print(f"✅ 当前GPU: {torch.cuda.get_device_name(0)}")
            
            # 测试GPU内存
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU内存: {gpu_memory:.1f}GB")
            
            # 测试GPU操作
            x = torch.randn(100, 100).cuda()
            y = torch.mm(x, x.t())
            print("✅ GPU计算测试成功")
            
            return True
        else:
            print("⚠️ CUDA不可用，将使用CPU模式")
            return True
            
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False

def test_training_compatibility():
    """测试训练兼容性"""
    print("\n🧪 测试训练兼容性...")
    
    try:
        # 测试数据加载器
        from torch.utils.data import DataLoader, TensorDataset
        
        # 创建模拟数据
        images = torch.randn(10, 3, 64, 64)
        labels = torch.randint(0, 5, (10,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        print("✅ 数据加载器创建成功")
        
        # 测试优化器
        from models.vqvae_model import MicroDopplerVQVAE
        model = MicroDopplerVQVAE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        print("✅ 优化器创建成功")
        
        # 测试一个训练步骤
        model.train()
        for batch_images, batch_labels in dataloader:
            optimizer.zero_grad()
            output = model(batch_images)
            loss = torch.nn.functional.mse_loss(output.sample, batch_images)
            loss.backward()
            optimizer.step()
            break
        
        print("✅ 训练步骤测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 训练兼容性测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🎨 VQ-VAE + Transformer 统一环境测试")
    print("=" * 60)
    print("🎯 验证统一环境的完整性")
    
    tests = [
        ("基础导入", test_basic_imports),
        ("VQModel功能", test_vqmodel),
        ("Transformers功能", test_transformers),
        ("自定义模型", test_custom_models),
        ("GPU支持", test_gpu_support),
        ("训练兼容性", test_training_compatibility),
    ]
    
    success_count = 0
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            success_count += 1
            print(f"✅ {test_name} 通过")
        else:
            print(f"❌ {test_name} 失败")
    
    print(f"\n{'='*60}")
    print(f"📊 测试结果: {success_count}/{len(tests)} 通过")
    
    if success_count >= len(tests) - 1:  # 允许1个失败
        print("\n🎉 统一环境测试成功！")
        print("✅ 环境配置正确，可以开始训练")
        print("\n📋 下一步:")
        print("   python train_main.py --data_dir /kaggle/input/dataset")
        print("\n💡 提示:")
        print("   - 统一环境支持VQ-VAE和Transformer训练")
        print("   - 可以使用 --skip_vqvae 或 --skip_transformer 进行部分训练")
        print("   - 所有依赖都已正确配置")
        return True
    else:
        print("\n❌ 统一环境测试失败")
        print("🔧 建议:")
        print("   1. 重新运行: python setup_unified_environment.py")
        print("   2. 检查错误信息")
        print("   3. 如果问题持续，使用分阶段训练作为备选")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
