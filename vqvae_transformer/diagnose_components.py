#!/usr/bin/env python3
"""
组件诊断脚本 - 分离测试VQ-VAE和Transformer
判断生成模式崩溃是哪个组件的问题
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import MicroDopplerDataset
from torch.utils.data import DataLoader

class ComponentDiagnostic:
    """组件诊断器"""
    
    def __init__(self, vqvae_path, transformer_path=None, data_dir="data/processed"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vqvae_path = Path(vqvae_path)
        self.transformer_path = Path(transformer_path) if transformer_path else None
        self.data_dir = Path(data_dir)
        
        print(f"🔍 组件诊断器初始化")
        print(f"   设备: {self.device}")
        print(f"   VQ-VAE路径: {self.vqvae_path}")
        print(f"   Transformer路径: {self.transformer_path}")
        
        # 加载模型
        self.vqvae_model = self._load_vqvae()
        self.transformer_model = self._load_transformer() if self.transformer_path else None
        
        # 加载测试数据
        self.test_data = self._load_test_data()
    
    def _load_vqvae(self):
        """加载VQ-VAE模型"""
        try:
            from diffusers import VQModel
            
            # 尝试加载diffusers格式
            if (self.vqvae_path / "config.json").exists():
                print("📁 加载diffusers格式VQ-VAE...")
                vqvae = VQModel.from_pretrained(str(self.vqvae_path))
            else:
                # 尝试加载checkpoint格式
                print("📁 加载checkpoint格式VQ-VAE...")
                checkpoint_files = list(self.vqvae_path.glob("*.pth"))
                if not checkpoint_files:
                    raise FileNotFoundError(f"未找到VQ-VAE模型文件: {self.vqvae_path}")
                
                checkpoint_path = checkpoint_files[0]
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # 创建VQ-VAE模型
                vqvae = VQModel(
                    in_channels=1,
                    out_channels=1,
                    down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                    up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                    block_out_channels=[128, 256],
                    layers_per_block=2,
                    act_fn="silu",
                    latent_channels=256,
                    sample_size=128,
                    num_vq_embeddings=1024,
                    vq_embed_dim=256,
                )
                
                vqvae.load_state_dict(checkpoint['model_state_dict'])
            
            vqvae.to(self.device)
            vqvae.eval()
            print("✅ VQ-VAE加载成功")
            return vqvae
            
        except Exception as e:
            print(f"❌ VQ-VAE加载失败: {e}")
            return None
    
    def _load_transformer(self):
        """加载Transformer模型"""
        try:
            from models.transformer_model import MicroDopplerTransformer
            
            checkpoint = torch.load(self.transformer_path, map_location=self.device)
            
            # 创建Transformer模型
            transformer = MicroDopplerTransformer(
                vocab_size=1024,
                max_seq_len=1024,
                num_users=31,
                d_model=256,
                nhead=8,
                num_layers=6,
                dim_feedforward=1024,
                dropout=0.1
            )
            
            transformer.load_state_dict(checkpoint['model_state_dict'])
            transformer.to(self.device)
            transformer.eval()
            print("✅ Transformer加载成功")
            return transformer
            
        except Exception as e:
            print(f"❌ Transformer加载失败: {e}")
            return None
    
    def _load_test_data(self):
        """加载测试数据"""
        try:
            dataset = MicroDopplerDataset(
                data_dir=str(self.data_dir),
                split='test',
                transform=None
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                num_workers=0
            )
            
            # 获取一个batch的测试数据
            test_batch = next(iter(dataloader))
            print(f"✅ 测试数据加载成功: {test_batch['image'].shape}")
            return test_batch
            
        except Exception as e:
            print(f"❌ 测试数据加载失败: {e}")
            return None
    
    def diagnose_vqvae(self):
        """诊断VQ-VAE组件"""
        print("\n" + "="*60)
        print("🔍 VQ-VAE组件诊断")
        print("="*60)
        
        if self.vqvae_model is None or self.test_data is None:
            print("❌ 无法进行VQ-VAE诊断：模型或数据未加载")
            return False
        
        images = self.test_data['image'].to(self.device)
        user_ids = self.test_data['user_id']
        
        with torch.no_grad():
            # 1. 编码测试
            print("1️⃣ 编码测试...")
            encoded = self.vqvae_model.encode(images)
            if hasattr(encoded, 'latents'):
                latents = encoded.latents
            else:
                latents = encoded
            print(f"   编码输出形状: {latents.shape}")
            
            # 2. 量化测试
            print("2️⃣ 量化测试...")
            quantized_output = self.vqvae_model.quantize(latents)
            if hasattr(quantized_output, 'quantized'):
                quantized = quantized_output.quantized
                indices = quantized_output.indices
            else:
                quantized = quantized_output
                indices = None
            
            print(f"   量化输出形状: {quantized.shape}")
            if indices is not None:
                print(f"   索引形状: {indices.shape}")
                print(f"   索引范围: [{indices.min().item()}, {indices.max().item()}]")
                
                # 分析码本使用
                unique_indices = torch.unique(indices)
                usage_ratio = len(unique_indices) / 1024
                print(f"   码本使用率: {len(unique_indices)}/1024 ({usage_ratio:.2%})")
            
            # 3. 解码测试
            print("3️⃣ 解码测试...")
            decoded = self.vqvae_model.decode(quantized)
            if hasattr(decoded, 'sample'):
                reconstructed = decoded.sample
            else:
                reconstructed = decoded
            print(f"   重建输出形状: {reconstructed.shape}")
            
            # 4. 重建质量评估
            print("4️⃣ 重建质量评估...")
            mse_loss = F.mse_loss(reconstructed, images)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_loss))
            
            print(f"   MSE损失: {mse_loss.item():.6f}")
            print(f"   PSNR: {psnr.item():.2f} dB")
            
            # 5. 用户特异性测试
            print("5️⃣ 用户特异性测试...")
            user_reconstructions = {}
            for i, uid in enumerate(user_ids):
                user_reconstructions[uid.item()] = reconstructed[i:i+1]
            
            if len(user_reconstructions) > 1:
                users = list(user_reconstructions.keys())
                img1 = user_reconstructions[users[0]]
                img2 = user_reconstructions[users[1]]
                user_diff = F.mse_loss(img1, img2)
                print(f"   用户间差异: {user_diff.item():.6f}")
            
            # 6. 保存诊断结果
            self._save_vqvae_diagnosis(images, reconstructed, user_ids)
            
            # 判断VQ-VAE质量
            vqvae_quality = self._evaluate_vqvae_quality(mse_loss.item(), usage_ratio, psnr.item())
            return vqvae_quality
    
    def diagnose_transformer(self):
        """诊断Transformer组件"""
        print("\n" + "="*60)
        print("🔍 Transformer组件诊断")
        print("="*60)
        
        if self.transformer_model is None or self.vqvae_model is None or self.test_data is None:
            print("❌ 无法进行Transformer诊断：模型或数据未加载")
            return False
        
        images = self.test_data['image'].to(self.device)
        user_ids = self.test_data['user_id'].to(self.device)
        
        with torch.no_grad():
            # 1. 获取真实tokens
            print("1️⃣ 获取真实tokens...")
            encoded = self.vqvae_model.encode(images)
            if hasattr(encoded, 'latents'):
                latents = encoded.latents
            else:
                latents = encoded
            
            quantized_output = self.vqvae_model.quantize(latents)
            if hasattr(quantized_output, 'indices'):
                real_tokens = quantized_output.indices.flatten(1)  # [B, H*W]
            else:
                print("❌ 无法获取token索引")
                return False
            
            print(f"   真实tokens形状: {real_tokens.shape}")
            print(f"   tokens范围: [{real_tokens.min().item()}, {real_tokens.max().item()}]")
            
            # 2. Transformer生成测试
            print("2️⃣ Transformer生成测试...")
            generated_tokens = self._generate_tokens(user_ids, max_length=real_tokens.shape[1])
            
            if generated_tokens is not None:
                print(f"   生成tokens形状: {generated_tokens.shape}")
                print(f"   生成tokens范围: [{generated_tokens.min().item()}, {generated_tokens.max().item()}]")
                
                # 3. Token分布分析
                print("3️⃣ Token分布分析...")
                self._analyze_token_distribution(real_tokens, generated_tokens)
                
                # 4. 用户条件测试
                print("4️⃣ 用户条件测试...")
                self._test_user_conditioning(user_ids)
                
                # 5. 生成图像质量测试
                print("5️⃣ 生成图像质量测试...")
                generated_images = self._decode_tokens(generated_tokens)
                if generated_images is not None:
                    self._save_transformer_diagnosis(images, generated_images, user_ids)
                    
                    # 判断Transformer质量
                    transformer_quality = self._evaluate_transformer_quality(real_tokens, generated_tokens, images, generated_images)
                    return transformer_quality
        
        return False
    
    def _generate_tokens(self, user_ids, max_length=1024):
        """生成tokens"""
        try:
            batch_size = user_ids.shape[0]
            generated = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
            
            for i in range(max_length):
                outputs = self.transformer_model(
                    input_ids=generated,
                    user_ids=user_ids
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if i % 100 == 0:
                    print(f"   生成进度: {i}/{max_length}")
            
            return generated[:, 1:]  # 移除起始token
            
        except Exception as e:
            print(f"❌ Token生成失败: {e}")
            return None
    
    def _decode_tokens(self, tokens):
        """解码tokens为图像"""
        try:
            # 重塑为2D
            batch_size = tokens.shape[0]
            tokens_2d = tokens.view(batch_size, 32, 32)  # 假设32x32
            
            # 创建量化特征
            quantized_features = self.vqvae_model.quantize.get_codebook_entry(
                tokens_2d.flatten(), tokens_2d.shape
            )
            
            # 解码
            decoded = self.vqvae_model.decode(quantized_features)
            if hasattr(decoded, 'sample'):
                return decoded.sample
            else:
                return decoded
                
        except Exception as e:
            print(f"❌ Token解码失败: {e}")
            return None
    
    def _analyze_token_distribution(self, real_tokens, generated_tokens):
        """分析token分布"""
        real_unique = torch.unique(real_tokens)
        gen_unique = torch.unique(generated_tokens)
        
        print(f"   真实tokens唯一值: {len(real_unique)}")
        print(f"   生成tokens唯一值: {len(gen_unique)}")
        
        # 计算分布差异
        real_hist = torch.histc(real_tokens.float(), bins=1024, min=0, max=1023)
        gen_hist = torch.histc(generated_tokens.float(), bins=1024, min=0, max=1023)
        
        # 归一化
        real_hist = real_hist / real_hist.sum()
        gen_hist = gen_hist / gen_hist.sum()
        
        # KL散度
        kl_div = F.kl_div(gen_hist.log(), real_hist, reduction='sum')
        print(f"   分布KL散度: {kl_div.item():.4f}")
    
    def _test_user_conditioning(self, user_ids):
        """测试用户条件"""
        if len(torch.unique(user_ids)) < 2:
            print("   ⚠️ 需要至少2个不同用户进行测试")
            return
        
        # 生成不同用户的tokens
        user1_id = user_ids[:1]
        user2_id = user_ids[1:2] if len(user_ids) > 1 else user_ids[:1]
        
        tokens1 = self._generate_tokens(user1_id, max_length=100)
        tokens2 = self._generate_tokens(user2_id, max_length=100)
        
        if tokens1 is not None and tokens2 is not None:
            # 计算差异
            diff_ratio = (tokens1 != tokens2).float().mean()
            print(f"   用户间token差异率: {diff_ratio.item():.2%}")
            
            if diff_ratio < 0.1:
                print("   ⚠️ 警告：用户间差异过小，可能存在模式崩溃")
    
    def _evaluate_vqvae_quality(self, mse_loss, usage_ratio, psnr):
        """评估VQ-VAE质量"""
        print("\n📊 VQ-VAE质量评估:")
        
        issues = []
        if mse_loss > 0.1:
            issues.append("重建误差过高")
        if usage_ratio < 0.1:
            issues.append("码本使用率过低")
        if psnr < 15:
            issues.append("PSNR过低")
        
        if not issues:
            print("   ✅ VQ-VAE质量良好")
            return True
        else:
            print("   ❌ VQ-VAE存在问题:")
            for issue in issues:
                print(f"      - {issue}")
            return False
    
    def _evaluate_transformer_quality(self, real_tokens, generated_tokens, real_images, generated_images):
        """评估Transformer质量"""
        print("\n📊 Transformer质量评估:")
        
        issues = []
        
        # Token多样性检查
        gen_unique_ratio = len(torch.unique(generated_tokens)) / 1024
        if gen_unique_ratio < 0.05:
            issues.append("生成token多样性不足")
        
        # 图像质量检查
        if generated_images is not None:
            img_mse = F.mse_loss(generated_images, real_images)
            if img_mse > 0.5:
                issues.append("生成图像质量差")
        
        if not issues:
            print("   ✅ Transformer质量良好")
            return True
        else:
            print("   ❌ Transformer存在问题:")
            for issue in issues:
                print(f"      - {issue}")
            return False
    
    def _save_vqvae_diagnosis(self, original, reconstructed, user_ids):
        """保存VQ-VAE诊断结果"""
        try:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            for i in range(min(4, original.shape[0])):
                # 原始图像
                axes[0, i].imshow(original[i, 0].cpu().numpy(), cmap='viridis')
                axes[0, i].set_title(f'原始 (User {user_ids[i].item()})')
                axes[0, i].axis('off')
                
                # 重建图像
                axes[1, i].imshow(reconstructed[i, 0].cpu().numpy(), cmap='viridis')
                axes[1, i].set_title(f'重建 (User {user_ids[i].item()})')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig('vqvae_diagnosis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("💾 VQ-VAE诊断图像已保存: vqvae_diagnosis.png")
            
        except Exception as e:
            print(f"⚠️ 保存VQ-VAE诊断图像失败: {e}")
    
    def _save_transformer_diagnosis(self, original, generated, user_ids):
        """保存Transformer诊断结果"""
        try:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            for i in range(min(4, original.shape[0])):
                # 原始图像
                axes[0, i].imshow(original[i, 0].cpu().numpy(), cmap='viridis')
                axes[0, i].set_title(f'真实 (User {user_ids[i].item()})')
                axes[0, i].axis('off')
                
                # 生成图像
                axes[1, i].imshow(generated[i, 0].cpu().numpy(), cmap='viridis')
                axes[1, i].set_title(f'生成 (User {user_ids[i].item()})')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig('transformer_diagnosis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("💾 Transformer诊断图像已保存: transformer_diagnosis.png")
            
        except Exception as e:
            print(f"⚠️ 保存Transformer诊断图像失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="组件诊断")
    parser.add_argument("--vqvae_path", type=str, default="models/vqvae_model", help="VQ-VAE模型路径")
    parser.add_argument("--transformer_path", type=str, help="Transformer模型路径")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="数据目录")
    
    args = parser.parse_args()
    
    # 创建诊断器
    diagnostic = ComponentDiagnostic(
        vqvae_path=args.vqvae_path,
        transformer_path=args.transformer_path,
        data_dir=args.data_dir
    )
    
    # 诊断VQ-VAE
    vqvae_ok = diagnostic.diagnose_vqvae()
    
    # 诊断Transformer（如果提供了路径）
    transformer_ok = True
    if args.transformer_path:
        transformer_ok = diagnostic.diagnose_transformer()
    
    # 总结
    print("\n" + "="*60)
    print("🎯 诊断总结")
    print("="*60)
    
    if vqvae_ok and transformer_ok:
        print("✅ 两个组件都正常，问题可能在训练策略或参数设置")
    elif not vqvae_ok and transformer_ok:
        print("❌ VQ-VAE存在问题，建议重新训练VQ-VAE")
    elif vqvae_ok and not transformer_ok:
        print("❌ Transformer存在问题，建议使用改进的训练脚本")
    else:
        print("❌ 两个组件都存在问题，建议从VQ-VAE开始重新训练")

if __name__ == "__main__":
    main()
