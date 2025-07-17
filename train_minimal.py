#!/usr/bin/env python3
"""
极简训练器 - 最小开销，最大效率
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/kaggle/working/VAE')

def setup_minimal_environment():
    """设置极简环境"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 单GPU
    os.environ['PYTHONUNBUFFERED'] = '1'
    torch.backends.cudnn.benchmark = True

def create_minimal_vae():
    """创建极简VAE"""
    from diffusers import AutoencoderKL
    
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=[
            "DownEncoderBlock2D",  # 256→128
            "DownEncoderBlock2D",  # 128→64
            "DownEncoderBlock2D"   # 64→32
        ],
        up_block_types=[
            "UpDecoderBlock2D",    # 32→64
            "UpDecoderBlock2D",    # 64→128
            "UpDecoderBlock2D"     # 128→256
        ],
        block_out_channels=[128, 256, 512],
        latent_channels=4,
        sample_size=256,
        layers_per_block=2,
    )
    
    return vae

def minimal_training():
    """极简训练循环"""
    
    setup_minimal_environment()
    
    print("🚀 极简高效训练")
    print("=" * 50)
    
    device = torch.device("cuda:0")
    
    # 1. 创建模型
    print("📦 创建VAE模型...")
    vae = create_minimal_vae()
    vae = vae.to(device)
    vae.train()
    
    # 启用混合精度
    scaler = torch.cuda.amp.GradScaler()
    
    # 2. 创建数据集
    print("📁 加载数据集...")
    from utils.data_loader import MicroDopplerDataset
    
    dataset = MicroDopplerDataset(
        data_dir="/kaggle/input/dataset",
        resolution=256,
        split="train"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,  # 大批次
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"   数据集大小: {len(dataset)}")
    print(f"   批次数: {len(dataloader)}")
    
    # 3. 创建优化器
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=2e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    
    # 4. 损失函数
    mse_loss = nn.MSELoss()
    
    # 5. 训练循环
    print("🚀 开始训练...")
    
    num_epochs = 20  # 减少epoch数
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            dynamic_ncols=True
        )
        
        epoch_loss = 0.0
        
        for step, batch in enumerate(progress_bar):
            step_start_time = time.time()
            
            # 获取数据
            images = batch['image'].to(device, non_blocking=True)
            
            # 前向传播 (混合精度)
            with torch.cuda.amp.autocast():
                posterior = vae.encode(images).latent_dist
                latents = posterior.sample()
                reconstruction = vae.decode(latents).sample
                
                # 计算损失
                recon_loss = mse_loss(reconstruction, images)
                kl_loss = posterior.kl().mean()
                total_loss = recon_loss + 1e-6 * kl_loss
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 统计
            epoch_loss += total_loss.item()
            global_step += 1
            
            step_time = time.time() - step_start_time
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'kl': f"{kl_loss.item():.6f}",
                'step_time': f"{step_time:.2f}s"
            })
            
            # 定期清理内存
            if step % 100 == 0:
                torch.cuda.empty_cache()
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(dataloader)
        
        print(f"Epoch {epoch+1} 完成:")
        print(f"   时间: {epoch_time/60:.1f}分钟")
        print(f"   平均损失: {avg_loss:.4f}")
        print(f"   预计剩余: {epoch_time * (num_epochs - epoch - 1) / 60:.1f}分钟")
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            save_path = f"/kaggle/working/vae_epoch_{epoch+1}.pth"
            torch.save(vae.state_dict(), save_path)
            print(f"   保存检查点: {save_path}")
    
    # 保存最终模型
    final_path = "/kaggle/working/vae_final.pth"
    torch.save(vae.state_dict(), final_path)
    print(f"✅ 训练完成! 模型保存: {final_path}")

def main():
    """主函数"""
    print("⚡ 极简高效VAE训练")
    print("=" * 50)
    
    print("🎯 极简策略:")
    print("   ✅ 单GPU避免通信开销")
    print("   ✅ 混合精度加速")
    print("   ✅ 大批次提高效率")
    print("   ✅ 去除不必要的框架")
    print("   ✅ 3层下采样 (55M参数)")
    
    print("\n📊 预期效果:")
    print("   🚀 训练速度: 最大化")
    print("   ⏱️  每轮时间: 10-15分钟")
    print("   💾 内存使用: ~6GB")
    print("   🎯 质量: 保持")
    
    try:
        minimal_training()
        print("\n🎉 极简训练成功!")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
