#!/usr/bin/env python3
"""
Transformer训练诊断工具
分析PSNR停滞问题，检查生成质量和token分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from models.vqvae_model import MicroDopplerVQVAE
from models.transformer_model import MicroDopplerTransformer
from utils.data_loader import MicroDopplerDataset

def load_models(vqvae_path, transformer_path):
    """加载VQ-VAE和Transformer模型"""
    print("📂 加载模型...")
    
    # 加载VQ-VAE
    vqvae = MicroDopplerVQVAE.from_pretrained(vqvae_path)
    vqvae.eval()
    print("✅ VQ-VAE加载成功")
    
    # 加载Transformer
    checkpoint = torch.load(transformer_path, map_location='cpu')
    
    transformer = MicroDopplerTransformer(
        vocab_size=checkpoint['args'].codebook_size,
        max_seq_len=1024,
        num_users=checkpoint['args'].num_users,
        n_embd=checkpoint['args'].n_embd,
        n_layer=checkpoint['args'].n_layer,
        n_head=checkpoint['args'].n_head,
        use_cross_attention=True,
    )
    
    transformer.load_state_dict(checkpoint['model_state_dict'])
    transformer.eval()
    print("✅ Transformer加载成功")
    
    return vqvae, transformer

def analyze_token_generation(transformer, vqvae, device, num_samples=10):
    """分析token生成质量"""
    print("🔍 分析token生成...")
    
    transformer.to(device)
    vqvae.to(device)
    
    # 生成多个用户的token
    user_ids = torch.tensor([1, 5, 10, 15, 20], device=device)
    
    generated_tokens_list = []
    
    for user_id in user_ids:
        print(f"   生成用户{user_id.item()}的token...")
        
        # 生成token序列
        with torch.no_grad():
            generated_tokens = generate_tokens(transformer, user_id.unsqueeze(0), device)
            
            if generated_tokens is not None:
                generated_tokens_list.append(generated_tokens[0].cpu().numpy())
                
                # 分析token分布
                unique_tokens = np.unique(generated_tokens[0].cpu().numpy())
                print(f"     唯一token数量: {len(unique_tokens)}")
                print(f"     token范围: [{unique_tokens.min()}, {unique_tokens.max()}]")
    
    # 分析token多样性
    if generated_tokens_list:
        analyze_token_diversity(generated_tokens_list)
    
    return generated_tokens_list

def generate_tokens(transformer, user_ids, device, max_length=1024):
    """生成token序列"""
    try:
        batch_size = user_ids.shape[0]
        
        # 开始token
        generated = torch.full((batch_size, 1), transformer.user_token_id, device=device)
        
        for step in range(max_length):
            # 准备输入
            inputs = transformer.prepare_inputs(user_ids, None)
            inputs['input_ids'] = generated
            inputs['attention_mask'] = torch.ones_like(generated)
            
            # 前向传播
            if transformer.use_cross_attention:
                outputs = transformer.transformer(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    encoder_hidden_states=inputs['encoder_hidden_states'],
                    encoder_attention_mask=inputs['encoder_attention_mask'],
                )
            else:
                outputs = transformer.transformer(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                )
            
            # 获取下一个token
            next_token_logits = outputs.logits[:, -1, :] / 1.0  # temperature=1.0
            
            # 限制到有效范围
            if next_token_logits.shape[-1] > 1024:
                next_token_logits = next_token_logits[:, :1024]
            
            # 采样
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
            next_token = torch.clamp(next_token, 0, 1023)
            
            # 添加到序列
            generated = torch.cat([generated, next_token], dim=1)
            
            if generated.shape[1] >= max_length + 1:
                break
        
        # 返回图像token（去掉用户token）
        return generated[:, 1:]
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return None

def analyze_token_diversity(token_lists):
    """分析token多样性"""
    print("📊 Token多样性分析:")
    
    # 计算每个序列的唯一token比例
    diversity_scores = []
    for tokens in token_lists:
        unique_ratio = len(np.unique(tokens)) / len(tokens)
        diversity_scores.append(unique_ratio)
    
    avg_diversity = np.mean(diversity_scores)
    print(f"   平均唯一token比例: {avg_diversity:.3f}")
    
    # 检查是否所有序列都相同
    if len(token_lists) > 1:
        all_same = all(np.array_equal(token_lists[0], tokens) for tokens in token_lists[1:])
        print(f"   所有序列是否相同: {'是' if all_same else '否'}")
    
    # 分析token分布
    all_tokens = np.concatenate(token_lists)
    unique_tokens = np.unique(all_tokens)
    print(f"   总体唯一token数量: {len(unique_tokens)} / 1024")
    print(f"   码本利用率: {len(unique_tokens)/1024*100:.1f}%")

def test_vqvae_reconstruction(vqvae, device, data_dir):
    """测试VQ-VAE重建质量"""
    print("🔍 测试VQ-VAE重建质量...")
    
    # 创建数据加载器
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = MicroDopplerDataset(data_dir, transform=transform, return_user_id=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    vqvae.to(device)
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            
            # VQ-VAE编码-解码
            encoded = vqvae.encode(images, return_dict=True)
            tokens = encoded['encoding_indices']
            
            print(f"   原始图像形状: {images.shape}")
            print(f"   Token形状: {tokens.shape}")
            print(f"   Token范围: [{tokens.min().item()}, {tokens.max().item()}]")
            
            # 重建
            reconstructed = vqvae.decode(encoded['latents'])
            
            # 计算PSNR
            mse = torch.mean((images - reconstructed) ** 2)
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # 范围[-1,1]，所以max=2
            
            print(f"   VQ-VAE重建PSNR: {psnr.item():.2f} dB")
            break

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transformer训练诊断")
    parser.add_argument("--vqvae_path", type=str, required=True, help="VQ-VAE模型路径")
    parser.add_argument("--transformer_path", type=str, required=True, help="Transformer模型路径")
    parser.add_argument("--data_dir", type=str, required=True, help="数据集路径")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用设备: {device}")
    
    try:
        # 1. 加载模型
        vqvae, transformer = load_models(args.vqvae_path, args.transformer_path)
        
        # 2. 测试VQ-VAE重建
        test_vqvae_reconstruction(vqvae, device, args.data_dir)
        
        # 3. 分析Transformer生成
        token_lists = analyze_token_generation(transformer, vqvae, device)
        
        # 4. 给出诊断结论
        print("\n🎯 诊断结论:")
        if not token_lists:
            print("❌ Transformer生成失败，可能存在严重问题")
        else:
            print("✅ Transformer能够生成token")
            print("💡 建议检查生成的图像质量和token多样性")
            
    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
