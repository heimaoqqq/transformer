#!/usr/bin/env python3
"""
快速组件检查 - 5分钟内判断是VQ-VAE还是Transformer的问题
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_vqvae_check(vqvae_path="models/vqvae_model", data_dir="data/processed"):
    """快速检查VQ-VAE质量"""
    print("🔍 快速VQ-VAE检查...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 1. 加载VQ-VAE
        from diffusers import VQModel
        
        if Path(vqvae_path + "/config.json").exists():
            vqvae = VQModel.from_pretrained(vqvae_path)
        else:
            print("❌ 未找到VQ-VAE模型")
            return False
        
        vqvae.to(device)
        vqvae.eval()
        
        # 2. 加载测试数据
        from vqvae_transformer.utils.data_loader import MicroDopplerDataset
        from torch.utils.data import DataLoader

        # 尝试不同的数据加载方式
        try:
            dataset = MicroDopplerDataset(data_dir=data_dir, split='test')
        except TypeError:
            try:
                dataset = MicroDopplerDataset(data_dir=data_dir)
            except Exception:
                dataset = MicroDopplerDataset(data_dir)

        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        test_batch = next(iter(dataloader))
        
        images = test_batch['image'].to(device)
        
        # 3. 快速重建测试
        with torch.no_grad():
            encoded = vqvae.encode(images)
            latents = encoded.latents if hasattr(encoded, 'latents') else encoded
            
            quantized_output = vqvae.quantize(latents)
            quantized = quantized_output.quantized if hasattr(quantized_output, 'quantized') else quantized_output
            indices = quantized_output.indices if hasattr(quantized_output, 'indices') else None
            
            decoded = vqvae.decode(quantized)
            reconstructed = decoded.sample if hasattr(decoded, 'sample') else decoded
            
            # 4. 关键指标
            mse_loss = F.mse_loss(reconstructed, images).item()
            psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse_loss))).item()
            
            if indices is not None:
                unique_tokens = len(torch.unique(indices))
                usage_ratio = unique_tokens / 1024
            else:
                usage_ratio = 0
            
            print(f"   MSE损失: {mse_loss:.6f}")
            print(f"   PSNR: {psnr:.2f} dB")
            print(f"   码本使用率: {usage_ratio:.2%}")
            
            # 5. 判断
            vqvae_issues = []
            if mse_loss > 0.1:
                vqvae_issues.append("重建误差过高")
            if psnr < 15:
                vqvae_issues.append("PSNR过低")
            if usage_ratio < 0.1:
                vqvae_issues.append("码本使用率过低")
            
            if vqvae_issues:
                print(f"   ❌ VQ-VAE问题: {', '.join(vqvae_issues)}")
                return False
            else:
                print("   ✅ VQ-VAE质量良好")
                return True
                
    except Exception as e:
        print(f"❌ VQ-VAE检查失败: {e}")
        return False

def quick_transformer_check(transformer_path, vqvae_path="models/vqvae_model", data_dir="data/processed"):
    """快速检查Transformer质量"""
    print("🔍 快速Transformer检查...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 1. 加载模型
        from diffusers import VQModel
        from vqvae_transformer.models.transformer_model import MicroDopplerTransformer
        
        # VQ-VAE
        vqvae = VQModel.from_pretrained(vqvae_path)
        vqvae.to(device)
        vqvae.eval()
        
        # Transformer - 修复PyTorch 2.6的weights_only问题
        checkpoint = torch.load(transformer_path, map_location=device, weights_only=False)
        transformer = MicroDopplerTransformer(
            vocab_size=1024,
            max_seq_len=1024,
            num_users=31,
            d_model=256,
            nhead=8,
            num_layers=6
        )
        transformer.load_state_dict(checkpoint['model_state_dict'])
        transformer.to(device)
        transformer.eval()
        
        # 2. 加载测试数据
        from vqvae_transformer.utils.data_loader import MicroDopplerDataset
        from torch.utils.data import DataLoader

        # 尝试不同的数据加载方式
        try:
            dataset = MicroDopplerDataset(data_dir=data_dir, split='test')
        except TypeError:
            try:
                dataset = MicroDopplerDataset(data_dir=data_dir)
            except Exception:
                dataset = MicroDopplerDataset(data_dir)

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        test_batch = next(iter(dataloader))
        
        images = test_batch['image'].to(device)
        user_ids = test_batch['user_id'].to(device)
        
        # 3. 快速生成测试
        with torch.no_grad():
            # 获取真实tokens
            encoded = vqvae.encode(images)
            latents = encoded.latents if hasattr(encoded, 'latents') else encoded
            quantized_output = vqvae.quantize(latents)
            real_tokens = quantized_output.indices.flatten(1) if hasattr(quantized_output, 'indices') else None
            
            if real_tokens is None:
                print("❌ 无法获取真实tokens")
                return False
            
            # 生成少量tokens进行测试
            batch_size = user_ids.shape[0]
            generated = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            
            for i in range(50):  # 只生成50个token进行快速测试
                outputs = transformer(input_ids=generated, user_ids=user_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
                generated = torch.cat([generated, next_token], dim=1)
            
            generated_tokens = generated[:, 1:]  # 移除起始token
            
            # 4. 关键指标
            gen_unique = len(torch.unique(generated_tokens))
            gen_diversity = gen_unique / (batch_size * 50)
            
            # 用户差异测试
            if batch_size > 1:
                user_diff = (generated_tokens[0] != generated_tokens[1]).float().mean().item()
            else:
                user_diff = 0.5  # 假设有差异
            
            print(f"   生成token唯一值: {gen_unique}")
            print(f"   生成多样性: {gen_diversity:.2%}")
            print(f"   用户间差异: {user_diff:.2%}")
            
            # 5. 判断
            transformer_issues = []
            if gen_diversity < 0.1:
                transformer_issues.append("生成多样性不足")
            if user_diff < 0.05:
                transformer_issues.append("用户条件无效")
            
            # 检查是否总是生成相同token
            if gen_unique < 5:
                transformer_issues.append("模式崩溃")
            
            if transformer_issues:
                print(f"   ❌ Transformer问题: {', '.join(transformer_issues)}")
                return False
            else:
                print("   ✅ Transformer质量良好")
                return True
                
    except Exception as e:
        print(f"❌ Transformer检查失败: {e}")
        return False

def main():
    """主函数"""
    print("🎯 遵循指南：快速组件诊断")
    print("="*50)
    
    import argparse
    parser = argparse.ArgumentParser(description="快速组件检查")
    parser.add_argument("--vqvae_path", type=str, default="models/vqvae_model", help="VQ-VAE路径")
    parser.add_argument("--transformer_path", type=str, help="Transformer路径")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="数据目录")
    
    args = parser.parse_args()
    
    # 1. 检查VQ-VAE
    vqvae_ok = quick_vqvae_check(args.vqvae_path, args.data_dir)
    
    # 2. 检查Transformer（如果提供）
    transformer_ok = True
    if args.transformer_path and Path(args.transformer_path).exists():
        transformer_ok = quick_transformer_check(args.transformer_path, args.vqvae_path, args.data_dir)
    elif args.transformer_path:
        print(f"⚠️ Transformer模型文件不存在: {args.transformer_path}")
        transformer_ok = False
    
    # 3. 诊断结论
    print("\n" + "="*50)
    print("🎯 快速诊断结论")
    print("="*50)
    
    if not vqvae_ok:
        print("❌ 问题源头：VQ-VAE")
        print("🔧 建议解决方案：")
        print("   1. 重新训练VQ-VAE，增加训练轮数")
        print("   2. 调整VQ-VAE的学习率和损失权重")
        print("   3. 检查数据预处理是否正确")
        print("   4. 考虑使用更大的码本或调整量化参数")
        
    elif not transformer_ok:
        print("❌ 问题源头：Transformer")
        print("🔧 建议解决方案：")
        print("   1. 使用 train_improved.py 重新训练")
        print("   2. 降低学习率，增加正则化")
        print("   3. 添加空间一致性损失")
        print("   4. 使用更保守的生成策略")
        
    else:
        print("✅ 两个组件都正常")
        print("🔧 可能的问题：")
        print("   1. 训练参数设置不当")
        print("   2. 数据质量问题")
        print("   3. 训练时间不足")
        print("   4. 生成参数需要调整")
    
    print("\n💡 下一步行动：")
    if not vqvae_ok:
        print("   优先修复VQ-VAE，然后再训练Transformer")
    elif not transformer_ok:
        print("   VQ-VAE正常，专注于改进Transformer训练")
    else:
        print("   两个组件都正常，检查整体训练流程")

if __name__ == "__main__":
    main()
