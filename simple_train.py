#!/usr/bin/env python3
"""
简化的训练脚本 - 逐步调试
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import time

# 设置无缓冲输出
os.environ['PYTHONUNBUFFERED'] = '1'

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

def simple_vae_training():
    """简化的VAE训练"""
    
    print("🚀 开始简化VAE训练")
    print("=" * 50)
    
    # 1. 检查GPU
    print("1️⃣ 检查GPU...")
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 个GPU")
    
    device = torch.device("cuda:0")
    
    # 2. 创建数据加载器
    print("\n2️⃣ 创建数据加载器...")
    try:
        from utils.data_loader import MicroDopplerDataset
        from torch.utils.data import DataLoader
        
        dataset = MicroDopplerDataset(
            data_dir="/kaggle/input/dataset",
            resolution=128,  # 使用更小的分辨率
            split="train"
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,  # 很小的批次
            shuffle=True,
            num_workers=0,  # 单线程
            pin_memory=False  # 禁用pin_memory
        )
        
        print(f"✅ 数据集大小: {len(dataset)}")
        print(f"✅ 批次数: {len(dataloader)}")
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return False
    
    # 3. 创建模型
    print("\n3️⃣ 创建模型...")
    try:
        from diffusers import AutoencoderKL
        
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[128, 256],
            latent_channels=4,
            sample_size=128,
        )
        
        vae = vae.to(device)
        vae.train()
        
        print("✅ VAE模型创建成功")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False
    
    # 4. 创建优化器
    print("\n4️⃣ 创建优化器...")
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    
    # 5. 测试训练循环
    print("\n5️⃣ 开始训练循环...")

    try:
        epoch = 1
        max_steps = 5  # 只训练5步

        print(f"   📊 数据加载器长度: {len(dataloader)}")
        print(f"   📊 最大步数: {max_steps}")

        # 先测试数据迭代器
        print("   🔄 测试数据迭代器...")
        data_iter = iter(dataloader)

        for step in range(max_steps):
            print(f"\n   🔄 步骤 {step+1}/{max_steps}")

            # 获取数据
            print("      📥 获取批次数据...")
            try:
                batch = next(data_iter)
                print(f"      ✅ 数据获取成功: {batch['image'].shape}")
            except StopIteration:
                print("      ⚠️  数据迭代器结束")
                break
            except Exception as e:
                print(f"      ❌ 数据获取失败: {e}")
                break

            # 移动到GPU
            print("      📤 移动数据到GPU...")
            images = batch['image'].to(device)
            print(f"      ✅ 图像移动成功: {images.shape}")

            # 前向传播
            print("      🔄 前向传播...")
            start_time = time.time()

            try:
                print("         🔄 VAE编码...")
                posterior = vae.encode(images).latent_dist
                latents = posterior.sample()
                print(f"         ✅ 编码完成: {latents.shape}")

                print("         🔄 VAE解码...")
                reconstruction = vae.decode(latents).sample
                print(f"         ✅ 解码完成: {reconstruction.shape}")

                forward_time = time.time() - start_time
                print(f"      ✅ 前向传播完成 ({forward_time:.2f}s)")

            except Exception as e:
                print(f"      ❌ 前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                return False

            # 计算损失
            print("      📊 计算损失...")
            try:
                recon_loss = mse_loss(reconstruction, images)
                kl_loss = posterior.kl().mean()
                total_loss = recon_loss + 1e-6 * kl_loss

                print(f"      ✅ 损失计算完成: {total_loss.item():.4f}")

            except Exception as e:
                print(f"      ❌ 损失计算失败: {e}")
                return False

            # 反向传播
            print("      🔄 反向传播...")
            try:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                print("      ✅ 反向传播完成")

            except Exception as e:
                print(f"      ❌ 反向传播失败: {e}")
                return False

            # 清理内存
            torch.cuda.empty_cache()
            print(f"      🧹 内存清理完成")
        
        print("\n🎉 简化训练完成!")
        return True
        
    except Exception as e:
        print(f"\n❌ 训练循环失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = simple_vae_training()
    
    if success:
        print("\n✅ 简化训练成功!")
        print("💡 可以尝试运行完整训练")
    else:
        print("\n❌ 简化训练失败!")
        print("💡 请检查错误信息")

if __name__ == "__main__":
    main()
