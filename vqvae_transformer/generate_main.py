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
    
    def __init__(self, model_dir, device="auto", vqvae_path=None, transformer_path=None):
        self.model_dir = Path(model_dir)
        self.vqvae_path = Path(vqvae_path) if vqvae_path else None
        self.transformer_path = Path(transformer_path) if transformer_path else None

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
        if self.vqvae_path:
            vqvae_path = self.vqvae_path
        else:
            vqvae_path = self.model_dir / "vqvae"

        print(f"📦 加载VQ-VAE: {vqvae_path}")

        # 尝试diffusers格式加载
        try:
            if (vqvae_path / "config.json").exists():
                print("   检测到diffusers格式，使用from_pretrained加载...")
                model = MicroDopplerVQVAE.from_pretrained(str(vqvae_path))
                model.to(self.device)
                model.eval()
                return model
        except Exception as e:
            print(f"   diffusers格式加载失败: {e}")

        # 尝试checkpoint格式加载
        checkpoint_path = None
        if vqvae_path.is_file() and vqvae_path.suffix == '.pth':
            checkpoint_path = vqvae_path
        else:
            for filename in ["best_model.pth", "final_model.pth", "model.pth"]:
                potential_path = vqvae_path / filename
                if potential_path.exists():
                    checkpoint_path = potential_path
                    break

        if checkpoint_path and checkpoint_path.exists():
            print(f"   使用checkpoint格式: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # 重建模型
            model = MicroDopplerVQVAE(
                num_vq_embeddings=checkpoint['args'].codebook_size,
                commitment_cost=checkpoint['args'].commitment_cost,
                ema_decay=getattr(checkpoint['args'], 'ema_decay', 0.99),
            )
            model.load_state_dict(checkpoint['model_state_dict'])

            model.to(self.device)
            model.eval()
            return model

        raise FileNotFoundError(f"未找到VQ-VAE模型: {vqvae_path}")
    
    def _load_transformer(self):
        """加载Transformer模型"""
        if self.transformer_path:
            # 使用指定的Transformer路径
            if self.transformer_path.is_file():
                checkpoint_path = self.transformer_path
            else:
                checkpoint_path = self.transformer_path / "best_model.pth"
                if not checkpoint_path.exists():
                    checkpoint_path = self.transformer_path / "final_model.pth"
        else:
            # 使用默认的目录结构
            transformer_path = self.model_dir / "transformer"
            checkpoint_path = transformer_path / "best_model.pth"
            if not checkpoint_path.exists():
                checkpoint_path = transformer_path / "final_model.pth"

        print(f"📦 加载Transformer: {checkpoint_path}")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"未找到Transformer模型: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 检查模型类型
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # 检查是否是VQ-VAE模型（错误的文件）
        vqvae_keys = ['encoder.conv_in.weight', 'decoder.conv_in.weight', 'quantize.embedding.weight']
        transformer_keys = ['transformer.transformer.wte.weight', 'user_encoder.user_embedding.weight']

        is_vqvae = any(key in state_dict for key in vqvae_keys)
        is_transformer = any(key in state_dict for key in transformer_keys)

        if is_vqvae and not is_transformer:
            raise ValueError(
                f"❌ 检测到VQ-VAE模型权重，但期望Transformer模型！\n"
                f"   文件路径: {checkpoint_path}\n"
                f"   请检查您的Transformer模型路径是否正确。\n"
                f"   当前文件包含的是VQ-VAE模型权重，不是Transformer权重。"
            )

        args = checkpoint.get('args')
        if args is None:
            raise ValueError(f"Checkpoint中缺少args信息: {checkpoint_path}")

        # 重建模型 - 添加默认值处理
        model = MicroDopplerTransformer(
            vocab_size=getattr(args, 'codebook_size', 1024),
            max_seq_len=getattr(args, 'max_seq_len', 1024),
            num_users=getattr(args, 'num_users', 31),  # 默认31个用户
            n_embd=getattr(args, 'n_embd', 512),
            n_layer=getattr(args, 'n_layer', 8),
            n_head=getattr(args, 'n_head', 8),
            dropout=getattr(args, 'dropout', 0.1),
            use_cross_attention=getattr(args, 'use_cross_attention', True),
        )

        model.load_state_dict(state_dict)
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
        # 🔧 修复：正确处理用户token和序列长度

        # 1. 移除用户token（如果存在）
        if len(tokens) == 1025:
            # 第一个token是用户token，移除它
            image_tokens = tokens[1:]
            print(f"   移除用户token，剩余图像token: {len(image_tokens)}")
        elif len(tokens) == 1024:
            # 已经是纯图像token
            image_tokens = tokens
        else:
            # 调整到1024个token
            if len(tokens) > 1024:
                image_tokens = tokens[:1024]
                print(f"   截断token序列到1024")
            else:
                # 填充到1024
                pad_tokens = torch.full((1024 - len(tokens),), 0, device=tokens.device)
                image_tokens = torch.cat([tokens, pad_tokens])
                print(f"   填充token序列到1024")

        # 2. 重塑为32x32（VQ-VAE的实际输出尺寸）
        tokens_2d = image_tokens.view(32, 32)

        # 3. 获取码本嵌入并解码
        with torch.no_grad():
            # 确保token索引在有效范围内
            tokens_2d = torch.clamp(tokens_2d, 0, self.vqvae_model.quantize.n_embed - 1)

            # 添加batch维度
            batch_tokens = tokens_2d.unsqueeze(0)  # [1, 32, 32]

            # 获取量化向量
            quantized_latents = self.vqvae_model.quantize.embedding(batch_tokens)  # [1, 32, 32, 256]
            quantized_latents = quantized_latents.permute(0, 3, 1, 2)  # [1, 256, 32, 32]

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
    parser.add_argument("--vqvae_path", type=str, default=None,
                       help="VQ-VAE模型路径（可选，如果不在默认位置）")
    parser.add_argument("--transformer_path", type=str, default=None,
                       help="Transformer模型路径（可选，如果不在默认位置）")
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
    generator = VQVAETransformerGenerator(
        model_dir=args.model_dir,
        device=args.device,
        vqvae_path=args.vqvae_path,
        transformer_path=args.transformer_path
    )
    
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
