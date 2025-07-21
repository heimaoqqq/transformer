#!/usr/bin/env python3
"""
VQ-VAE + Transformer 主生成脚本
从用户ID生成微多普勒时频图
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 条件导入模型 - 检查环境兼容性
try:
    from models.vqvae_model import MicroDopplerVQVAE
    VQVAE_AVAILABLE = True
except ImportError as e:
    print(f"❌ 无法导入VQ-VAE模型: {e}")
    print("   请确保在正确的环境中运行生成脚本")
    VQVAE_AVAILABLE = False

try:
    from models.transformer_model import MicroDopplerTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError as e:
    print(f"❌ 无法导入Transformer模型: {e}")
    print("   请确保在正确的环境中运行生成脚本")
    TRANSFORMER_AVAILABLE = False

# 检查必要组件
if not (VQVAE_AVAILABLE and TRANSFORMER_AVAILABLE):
    print("❌ 生成脚本需要同时支持VQ-VAE和Transformer")
    print("   建议在Transformer环境中运行，因为它可以加载VQ-VAE模型")
    sys.exit(1)

class VQVAETransformerGenerator:
    """VQ-VAE + Transformer 生成器"""
    
    def __init__(self, model_dir, device="auto"):
        self.model_dir = Path(model_dir)
        
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🎮 使用设备: {self.device}")
        
        # 加载模型
        self.vqvae_model = self._load_vqvae()
        self.transformer_model = self._load_transformer()
        
        print(f"✅ 模型加载完成")
    
    def _load_vqvae(self):
        """加载VQ-VAE模型"""
        vqvae_path = self.model_dir / "vqvae"
        
        print(f"📦 加载VQ-VAE: {vqvae_path}")
        
        checkpoint_path = vqvae_path / "best_model.pth"
        if not checkpoint_path.exists():
            checkpoint_path = vqvae_path / "final_model.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"未找到VQ-VAE模型: {vqvae_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # 重建模型
        model = MicroDopplerVQVAE(
            num_vq_embeddings=checkpoint['args'].codebook_size,
            commitment_cost=checkpoint['args'].commitment_cost,
            ema_decay=checkpoint['args'].ema_decay,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_transformer(self):
        """加载Transformer模型"""
        transformer_path = self.model_dir / "transformer"
        
        print(f"📦 加载Transformer: {transformer_path}")
        
        checkpoint_path = transformer_path / "best_model.pth"
        if not checkpoint_path.exists():
            checkpoint_path = transformer_path / "final_model.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"未找到Transformer模型: {transformer_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        args = checkpoint['args']
        
        # 重建模型
        model = MicroDopplerTransformer(
            vocab_size=args.codebook_size,
            max_seq_len=getattr(args, 'max_seq_len', 256),
            num_users=args.num_users,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            dropout=args.dropout,
            use_cross_attention=args.use_cross_attention,
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def generate_for_user(
        self,
        user_id: int,
        num_samples: int = 5,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        diversity_boost: float = 1.2,
    ):
        """
        为指定用户生成图像
        Args:
            user_id: 用户ID (0-30)
            num_samples: 生成样本数量
            temperature: 采样温度
            top_k: top-k采样
            top_p: nucleus采样
            diversity_boost: 多样性增强因子
        Returns:
            generated_images: [num_samples, 3, H, W] 生成的图像
        """
        print(f"🎨 为用户 {user_id} 生成 {num_samples} 张图像...")
        
        generated_images = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # 为每个样本使用略微不同的采样参数以增加多样性
                sample_temp = temperature * (1 + (i / num_samples - 0.5) * 0.2 * diversity_boost)
                sample_top_p = max(0.7, top_p - (i / num_samples) * 0.1 * diversity_boost)
                
                # 生成token序列
                user_ids = torch.tensor([user_id], device=self.device)
                
                generated_tokens = self.transformer_model.generate(
                    user_ids=user_ids,
                    max_length=self.transformer_model.max_seq_len,
                    temperature=sample_temp,
                    top_k=top_k,
                    top_p=sample_top_p,
                    do_sample=True,
                    num_return_sequences=1,
                )
                
                # 转换为图像
                image = self._tokens_to_image(generated_tokens[0])
                generated_images.append(image)
                
                print(f"  ✅ 样本 {i+1}/{num_samples} 完成")
        
        return torch.stack(generated_images)
    
    def _tokens_to_image(self, tokens):
        """将token序列转换为图像"""
        # 重塑为2D
        seq_len = len(tokens)
        latent_size = int(np.sqrt(seq_len))
        
        if latent_size * latent_size != seq_len:
            # 如果不是完全平方数，截断或填充
            target_len = latent_size * latent_size
            if seq_len > target_len:
                tokens = tokens[:target_len]
            else:
                pad_tokens = torch.full((target_len - seq_len,), 0, device=tokens.device)
                tokens = torch.cat([tokens, pad_tokens])
        
        # 重塑为2D
        tokens_2d = tokens.view(1, latent_size, latent_size)
        
        # 获取码本嵌入
        with torch.no_grad():
            # 确保token索引在有效范围内
            tokens_2d = torch.clamp(tokens_2d, 0, self.vqvae_model.quantize.n_embed - 1)
            
            # 获取量化向量
            quantized_latents = self.vqvae_model.quantize.embedding(tokens_2d)
            quantized_latents = quantized_latents.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            # 解码为图像
            generated_image = self.vqvae_model.decode(quantized_latents, force_not_quantize=True)
            
            # 归一化到[0,1]
            image = (generated_image.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
            
            return image
    
    def generate_dataset(
        self,
        output_dir: str,
        samples_per_user: int = 10,
        user_list: list = None,
        **generation_kwargs
    ):
        """
        生成完整数据集
        Args:
            output_dir: 输出目录
            samples_per_user: 每个用户生成的样本数
            user_list: 用户列表，None表示所有用户
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if user_list is None:
            user_list = list(range(self.transformer_model.num_users))
        
        print(f"🎯 生成数据集:")
        print(f"   输出目录: {output_path}")
        print(f"   用户数量: {len(user_list)}")
        print(f"   每用户样本数: {samples_per_user}")
        print(f"   总样本数: {len(user_list) * samples_per_user}")
        
        total_generated = 0
        
        for user_id in tqdm(user_list, desc="生成用户数据"):
            user_dir = output_path / f"user_{user_id:02d}"
            user_dir.mkdir(exist_ok=True)
            
            # 生成图像
            generated_images = self.generate_for_user(
                user_id=user_id,
                num_samples=samples_per_user,
                **generation_kwargs
            )
            
            # 保存图像
            for i, image in enumerate(generated_images):
                image_pil = transforms.ToPILImage()(image.cpu())
                save_path = user_dir / f"generated_{i:03d}.png"
                image_pil.save(save_path)
                total_generated += 1
            
            print(f"✅ 用户 {user_id}: {samples_per_user} 张图像保存到 {user_dir}")
        
        print(f"\n🎉 数据集生成完成!")
        print(f"   总计生成: {total_generated} 张图像")
        print(f"   保存位置: {output_path}")
        
        return output_path
    
    def visualize_generation(self, user_id: int, num_samples: int = 4, save_path: str = None):
        """可视化生成结果"""
        generated_images = self.generate_for_user(user_id, num_samples)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 4, 4))
        if num_samples == 1:
            axes = [axes]
        
        for i, image in enumerate(generated_images):
            axes[i].imshow(image.cpu().permute(1, 2, 0))
            axes[i].set_title(f'User {user_id} - Sample {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 可视化结果保存到: {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="VQ-VAE + Transformer 图像生成")
    
    # 模型参数
    parser.add_argument("--model_dir", type=str, required=True,
                       help="模型目录路径")
    parser.add_argument("--output_dir", type=str, default="generated_images",
                       help="输出目录")
    
    # 生成参数
    parser.add_argument("--user_id", type=int, default=None,
                       help="指定用户ID (不指定则生成所有用户)")
    parser.add_argument("--samples_per_user", type=int, default=10,
                       help="每个用户生成的样本数")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="采样温度")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k采样")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p采样")
    parser.add_argument("--diversity_boost", type=float, default=1.2,
                       help="多样性增强因子")
    
    # 功能选项
    parser.add_argument("--visualize_only", action="store_true",
                       help="只可视化，不保存数据集")
    parser.add_argument("--device", type=str, default="auto",
                       help="计算设备")
    
    args = parser.parse_args()
    
    print("🎨 VQ-VAE + Transformer 图像生成器")
    print("=" * 50)
    
    # 创建生成器
    generator = VQVAETransformerGenerator(args.model_dir, args.device)
    
    if args.visualize_only:
        # 只可视化
        user_id = args.user_id if args.user_id is not None else 0
        save_path = f"visualization_user_{user_id}.png"
        generator.visualize_generation(user_id, 4, save_path)
    else:
        # 生成数据集
        user_list = [args.user_id] if args.user_id is not None else None
        
        output_path = generator.generate_dataset(
            output_dir=args.output_dir,
            samples_per_user=args.samples_per_user,
            user_list=user_list,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            diversity_boost=args.diversity_boost,
        )
        
        print(f"\n🔍 下一步: 运行验证")
        print(f"   python validate_main.py \\")
        print(f"     --model_dir {args.model_dir} \\")
        print(f"     --real_data_dir /path/to/real/data \\")
        print(f"     --generated_data_dir {output_path} \\")
        print(f"     --target_user_id {args.user_id or 0}")

if __name__ == "__main__":
    main()
