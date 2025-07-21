#!/usr/bin/env python3
"""
简单的模型测试脚本
验证VQ-VAE和Transformer模型的基本功能
"""

import torch
from models.vqvae_model import MicroDopplerVQVAE
from models.transformer_model import MicroDopplerTransformer

def test_vqvae():
    """测试VQ-VAE模型"""
    print('🧪 测试VQ-VAE模型...')
    
    # 创建模型
    vqvae = MicroDopplerVQVAE()
    
    # 创建测试数据
    x = torch.randn(2, 3, 128, 128)
    
    # 前向传播
    with torch.no_grad():
        output = vqvae(x)
        print(f'✅ VQ-VAE: {x.shape} -> {output.sample.shape}')
        print(f'✅ VQ损失: {output.vq_loss.item():.4f}')
    
    return True

def test_transformer():
    """测试Transformer模型"""
    print('🧪 测试Transformer模型...')
    
    # 创建模型
    transformer = MicroDopplerTransformer()
    
    # 创建测试数据
    tokens = torch.randint(0, 1024, (2, 100))
    
    # 前向传播
    with torch.no_grad():
        output = transformer(tokens)
        print(f'✅ Transformer: {tokens.shape} -> {output.logits.shape}')
    
    return True

def main():
    """主函数"""
    print('🎨 简单模型测试')
    print('=' * 40)
    
    try:
        # 测试VQ-VAE
        test_vqvae()
        print()
        
        # 测试Transformer
        test_transformer()
        print()
        
        print('🎉 所有模型测试通过！')
        return True
        
    except Exception as e:
        print(f'❌ 测试失败: {e}')
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print('✅ 模型功能正常，可以开始训练！')
    else:
        print('❌ 模型测试失败，请检查代码')
