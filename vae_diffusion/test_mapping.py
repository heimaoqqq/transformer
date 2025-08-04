#!/usr/bin/env python3
"""
测试不同用户映射方式的脚本
用于确定训练时实际使用的映射逻辑
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List
from tqdm import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
import sys

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入UserConditionEncoder
import torch.nn as nn

class UserConditionEncoder(nn.Module):
    """用户条件编码器"""
    def __init__(self, num_users: int, embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.num_users = num_users
        self.embed_dim = embed_dim
        
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        
    def forward(self, user_indices):
        user_embeds = self.user_embedding(user_indices)
        user_embeds = self.mlp(user_embeds)
        return user_embeds

def test_mapping(
    vae_path: str,
    unet_path: str,
    condition_encoder_path: str,
    data_dir: str,
    test_user_id: int = 6,  # 使用已知效果好的用户6
    device: str = "auto"
):
    """测试不同映射方式的生成效果"""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🧪 测试用户映射方式")
    print(f"🎯 测试用户: {test_user_id}")
    print(f"🚀 设备: {device}")
    
    # 获取用户列表
    data_path = Path(data_dir)
    all_users = []
    for user_dir in data_path.iterdir():
        if user_dir.is_dir() and user_dir.name.startswith('ID_'):
            try:
                user_id = int(user_dir.name.split('_')[1])
                all_users.append(user_id)
            except ValueError:
                continue
    all_users = sorted(all_users)
    
    print(f"📊 所有用户: {all_users}")
    
    # 测试不同映射方式
    mappings = {
        "all_users.index": all_users.index(test_user_id) if test_user_id in all_users else None,
        "user_id - 1": test_user_id - 1 if test_user_id > 0 else test_user_id,
        "user_id": test_user_id,
        "user_id - 2": test_user_id - 2 if test_user_id > 1 else 0,
    }
    
    print(f"\n🗺️  映射方式测试:")
    for name, idx in mappings.items():
        if idx is not None:
            print(f"  {name:15s}: 用户{test_user_id} → 索引{idx}")
    
    # 加载模型
    print(f"\n📦 加载模型...")
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.to(device).eval()
    
    unet = UNet2DConditionModel.from_pretrained(unet_path)
    unet.to(device).eval()
    
    condition_encoder = UserConditionEncoder(
        num_users=len(all_users),
        embed_dim=unet.config.cross_attention_dim
    )
    condition_encoder_state = torch.load(condition_encoder_path, map_location='cpu')
    condition_encoder.load_state_dict(condition_encoder_state)
    condition_encoder.to(device).eval()
    
    # 调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="epsilon",
    )
    ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    ddim_scheduler.set_timesteps(20)  # 训练时默认
    
    # 测试生成
    output_dir = Path("./mapping_test_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🎨 开始测试生成...")
    
    with torch.no_grad():
        for mapping_name, user_idx in mappings.items():
            if user_idx is None or user_idx < 0 or user_idx >= len(all_users):
                print(f"⚠️  跳过无效映射: {mapping_name}")
                continue
                
            print(f"\n🧪 测试映射: {mapping_name} (索引{user_idx})")
            
            # 生成图像
            latents = torch.randn(1, 4, 32, 32, device=device)
            
            user_idx_tensor = torch.tensor([user_idx], device=device)
            encoder_hidden_states = condition_encoder(user_idx_tensor)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            
            # 去噪
            for t in tqdm(ddim_scheduler.timesteps, desc=f"生成 {mapping_name}", leave=False):
                timestep = t.unsqueeze(0).to(device)
                noise_pred = unet(
                    latents,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False
                )[0]
                latents = ddim_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # VAE解码
            vae_model = vae.module if hasattr(vae, 'module') else vae
            latents = latents / vae_model.config.scaling_factor
            image = vae_model.decode(latents).sample
            
            # 转换为PIL
            image = image.clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
            
            # 添加标签
            try:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(pil_image)
                draw.text((5, 5), f"User{test_user_id} {mapping_name}", fill=(255, 255, 255))
                draw.text((5, 20), f"Index: {user_idx}", fill=(255, 255, 255))
            except:
                pass
            
            # 保存
            filename = f"user{test_user_id}_{mapping_name.replace(' ', '_').replace('.', '_')}_idx{user_idx}.png"
            pil_image.save(output_dir / filename)
            print(f"  ✅ 保存: {filename}")
    
    print(f"\n🎉 测试完成！")
    print(f"📁 结果保存在: {output_dir}")
    print(f"💡 请查看生成的图像，找出与训练时可视化最相似的映射方式")

def main():
    parser = argparse.ArgumentParser(description="测试用户映射方式")
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--unet_path", type=str, required=True)
    parser.add_argument("--condition_encoder_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--test_user_id", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    test_mapping(
        vae_path=args.vae_path,
        unet_path=args.unet_path,
        condition_encoder_path=args.condition_encoder_path,
        data_dir=args.data_dir,
        test_user_id=args.test_user_id,
        device=args.device
    )

if __name__ == "__main__":
    main()
