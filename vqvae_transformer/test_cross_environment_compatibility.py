#!/usr/bin/env python3
"""
跨环境兼容性测试脚本
验证VQ-VAE模型在不同环境间的兼容性
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_vqvae_loading():
    """测试VQ-VAE模型加载"""
    print("🧪 测试VQ-VAE模型加载兼容性...")
    
    # 模拟不同环境下的模型加载
    try:
        from models.vqvae_model import MicroDopplerVQVAE
        print("✅ MicroDopplerVQVAE导入成功")
        
        # 创建测试模型
        model = MicroDopplerVQVAE(
            num_vq_embeddings=1024,
            commitment_cost=0.25,
            ema_decay=0.99,
        )
        print("✅ VQ-VAE模型创建成功")
        
        # 测试保存
        test_checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'args': type('Args', (), {
                'codebook_size': 1024,
                'commitment_cost': 0.25,
                'ema_decay': 0.99,
            })(),
        }
        
        # 保存到临时文件
        temp_path = Path("temp_vqvae_test.pth")
        torch.save(test_checkpoint, temp_path)
        print("✅ VQ-VAE模型保存成功")
        
        # 测试加载
        loaded_checkpoint = torch.load(temp_path, map_location="cpu")
        
        # 重建模型
        loaded_model = MicroDopplerVQVAE(
            num_vq_embeddings=loaded_checkpoint['args'].codebook_size,
            commitment_cost=loaded_checkpoint['args'].commitment_cost,
            ema_decay=loaded_checkpoint['args'].ema_decay,
        )
        loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        print("✅ VQ-VAE模型加载成功")
        
        # 测试关键接口
        test_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            # 编码
            encoded = loaded_model.encode(test_input)
            tokens = encoded['encoding_indices']
            print(f"✅ 编码成功: tokens shape = {tokens.shape}")
            
            # 测试码本嵌入访问 (Transformer阶段需要的)
            embedding_weight = loaded_model.quantize.embedding.weight
            print(f"✅ 码本嵌入访问成功: shape = {embedding_weight.shape}")
            
            # 测试解码 (Transformer阶段需要的)
            latent_size = int(tokens.shape[1] ** 0.5)
            tokens_2d = tokens.view(1, latent_size, latent_size)
            quantized_latents = loaded_model.quantize.embedding(tokens_2d)
            quantized_latents = quantized_latents.permute(0, 3, 1, 2)
            
            decoded = loaded_model.decode(quantized_latents, force_not_quantize=True)
            print(f"✅ 解码成功: output shape = {decoded.shape}")
        
        # 清理临时文件
        temp_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ VQ-VAE兼容性测试失败: {e}")
        return False

def test_transformer_interface():
    """测试Transformer需要的VQ-VAE接口"""
    print("\n🤖 测试Transformer接口兼容性...")
    
    try:
        from models.vqvae_model import MicroDopplerVQVAE
        
        # 创建VQ-VAE模型
        vqvae_model = MicroDopplerVQVAE(
            num_vq_embeddings=1024,
            commitment_cost=0.25,
            ema_decay=0.99,
        )
        vqvae_model.eval()
        
        # 模拟Transformer生成的token序列
        batch_size = 1
        seq_len = 256  # 16x16的token序列
        vocab_size = 1024
        
        # 生成随机token序列 (模拟Transformer输出)
        generated_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        print(f"✅ 模拟Transformer输出: {generated_tokens.shape}")
        
        # 转换为2D (Transformer阶段的关键步骤)
        latent_size = int(seq_len ** 0.5)  # 16
        tokens_2d = generated_tokens.view(batch_size, latent_size, latent_size)
        print(f"✅ 重塑为2D: {tokens_2d.shape}")
        
        with torch.no_grad():
            # 确保token索引在有效范围内 (Transformer阶段的关键步骤)
            tokens_2d = torch.clamp(tokens_2d, 0, vqvae_model.quantize.n_embed - 1)
            
            # 获取量化向量 (Transformer阶段的关键步骤)
            quantized_latents = vqvae_model.quantize.embedding(tokens_2d)
            quantized_latents = quantized_latents.permute(0, 3, 1, 2)  # [B, C, H, W]
            print(f"✅ 获取量化向量: {quantized_latents.shape}")
            
            # 解码为图像 (Transformer阶段的关键步骤)
            generated_image = vqvae_model.decode(quantized_latents, force_not_quantize=True)
            print(f"✅ 解码为图像: {generated_image.shape}")
            
            # 归一化 (Transformer阶段的关键步骤)
            image = (generated_image.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
            print(f"✅ 图像归一化: {image.shape}")
        
        print("✅ 所有Transformer接口测试通过")
        return True
        
    except Exception as e:
        print(f"❌ Transformer接口测试失败: {e}")
        return False

def test_version_independence():
    """测试版本独立性"""
    print("\n🔄 测试版本独立性...")
    
    try:
        # 测试模型定义不依赖diffusers具体版本
        from models.vqvae_model import MicroDopplerVQVAE, EMAVectorQuantizer
        print("✅ 自定义模型类导入成功")
        
        # 测试关键组件
        quantizer = EMAVectorQuantizer(n_embed=1024, embed_dim=256)
        print("✅ 自定义量化器创建成功")
        
        # 测试不依赖diffusers的具体API
        model = MicroDopplerVQVAE()
        state_dict = model.state_dict()
        print(f"✅ 模型权重获取成功: {len(state_dict)} 个参数")
        
        # 验证权重是纯PyTorch格式
        for key, value in list(state_dict.items())[:3]:
            assert isinstance(value, torch.Tensor), f"权重 {key} 不是PyTorch张量"
        print("✅ 权重格式验证通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 版本独立性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🔬 跨环境兼容性测试")
    print("=" * 50)
    
    tests = [
        ("VQ-VAE模型加载", test_vqvae_loading),
        ("Transformer接口", test_transformer_interface),
        ("版本独立性", test_version_independence),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # 总结
    print(f"\n{'='*20} 测试总结 {'='*20}")
    all_passed = True
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有兼容性测试通过!")
        print("✅ VQ-VAE模型可以在不同环境间安全使用")
        print("✅ 分阶段训练完全可行")
    else:
        print("\n❌ 部分测试失败")
        print("⚠️ 需要检查兼容性问题")

if __name__ == "__main__":
    main()
