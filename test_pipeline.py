#!/usr/bin/env python3
"""
微多普勒时频图数据增广项目 - 流程测试脚本
用于验证整个训练和生成流程
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import shutil

def create_test_data(output_dir: str, num_users: int = 5, images_per_user: int = 10):
    """创建测试数据"""
    print(f"Creating test data in {output_dir}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 为每个用户创建目录和图像 (使用ID_格式)
    for user_id in range(1, num_users + 1):
        user_dir = output_path / f"ID_{user_id}"
        user_dir.mkdir(exist_ok=True)
        
        for img_id in range(images_per_user):
            # 创建模拟的微多普勒时频图
            # 使用不同的颜色模式来模拟不同用户的特征
            img = create_mock_micro_doppler(user_id, img_id)
            img.save(user_dir / f"image_{img_id:03d}.png")
    
    print(f"Created test data for {num_users} users, {images_per_user} images each")

def create_mock_micro_doppler(user_id: int, img_id: int, size: int = 256):
    """创建模拟的微多普勒时频图"""
    # 创建基础噪声
    np.random.seed(user_id * 1000 + img_id)
    
    # 时间轴和频率轴
    t = np.linspace(0, 2*np.pi, size)
    f = np.linspace(0, 2*np.pi, size)
    T, F = np.meshgrid(t, f)
    
    # 为不同用户创建不同的频率模式
    # 模拟步态的周期性特征
    base_freq = 0.5 + user_id * 0.1  # 不同用户的基础频率
    
    # 主要信号：步态基频
    signal1 = np.sin(base_freq * T) * np.exp(-0.1 * F)
    
    # 谐波：步态的倍频成分
    signal2 = 0.5 * np.sin(2 * base_freq * T) * np.exp(-0.2 * F)
    signal3 = 0.3 * np.sin(3 * base_freq * T) * np.exp(-0.3 * F)
    
    # 微多普勒调制：肢体摆动
    modulation = 0.2 * np.sin(4 * base_freq * T + user_id) * np.cos(0.5 * F)
    
    # 组合信号
    combined = signal1 + signal2 + signal3 + modulation
    
    # 添加噪声
    noise = 0.1 * np.random.randn(size, size)
    combined += noise
    
    # 归一化到[0, 1]
    combined = (combined - combined.min()) / (combined.max() - combined.min())
    
    # 转换为RGB图像 (模拟时频图的伪彩色)
    # 使用jet colormap的简化版本
    rgb_image = np.zeros((size, size, 3))
    
    # 红色通道：高频成分
    rgb_image[:, :, 0] = combined
    
    # 绿色通道：中频成分
    rgb_image[:, :, 1] = np.roll(combined, size//4, axis=1)
    
    # 蓝色通道：低频成分
    rgb_image[:, :, 2] = np.roll(combined, size//2, axis=1)
    
    # 转换为PIL图像
    rgb_image = (rgb_image * 255).astype(np.uint8)
    return Image.fromarray(rgb_image)

def test_data_loader():
    """测试数据加载器"""
    print("\n=== Testing Data Loader ===")
    
    try:
        from utils.data_loader import MicroDopplerDataset, MicroDopplerDataModule
        from torch.utils.data import DataLoader
        
        # 测试数据集
        dataset = MicroDopplerDataset(
            data_dir="./test_data",
            resolution=256,
            augment=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of users: {dataset.num_users}")
        print(f"User mapping: {dataset.user_to_idx}")
        
        # 测试数据加载器
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        for batch in dataloader:
            print(f"Batch image shape: {batch['image'].shape}")
            print(f"Batch user IDs: {batch['user_id']}")
            print(f"Batch user indices: {batch['user_idx']}")
            break
        
        print("✅ Data loader test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_vae_model():
    """测试VAE模型"""
    print("\n=== Testing VAE Model ===")
    
    try:
        from diffusers import AutoencoderKL
        
        # 创建VAE模型
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            sample_size=256,
        )
        
        # 测试前向传播
        test_input = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            # 编码
            posterior = vae.encode(test_input).latent_dist
            latents = posterior.sample()
            
            # 解码
            reconstruction = vae.decode(latents).sample
        
        print(f"Input shape: {test_input.shape}")
        print(f"Latent shape: {latents.shape}")
        print(f"Reconstruction shape: {reconstruction.shape}")
        
        print("✅ VAE model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ VAE model test failed: {e}")
        return False

def test_unet_model():
    """测试UNet模型"""
    print("\n=== Testing UNet Model ===")
    
    try:
        from diffusers import UNet2DConditionModel
        
        # 创建UNet模型
        unet = UNet2DConditionModel(
            sample_size=32,  # 256/8 = 32
            in_channels=4,
            out_channels=4,
            cross_attention_dim=768,
        )
        
        # 测试前向传播
        test_latents = torch.randn(1, 4, 32, 32)
        test_timesteps = torch.randint(0, 1000, (1,))
        test_conditions = torch.randn(1, 1, 768)
        
        with torch.no_grad():
            noise_pred = unet(
                test_latents,
                test_timesteps,
                encoder_hidden_states=test_conditions
            ).sample
        
        print(f"Input latents shape: {test_latents.shape}")
        print(f"Timesteps shape: {test_timesteps.shape}")
        print(f"Conditions shape: {test_conditions.shape}")
        print(f"Noise prediction shape: {noise_pred.shape}")
        
        print("✅ UNet model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ UNet model test failed: {e}")
        return False

def test_condition_encoder():
    """测试条件编码器"""
    print("\n=== Testing Condition Encoder ===")
    
    try:
        import sys
        sys.path.append('./training')
        from train_diffusion import UserConditionEncoder
        
        # 创建条件编码器
        encoder = UserConditionEncoder(num_users=5, embed_dim=768)
        
        # 测试编码
        user_indices = torch.tensor([0, 1, 2, 3, 4])
        
        with torch.no_grad():
            embeddings = encoder(user_indices)
        
        print(f"User indices: {user_indices}")
        print(f"Embeddings shape: {embeddings.shape}")
        
        print("✅ Condition encoder test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Condition encoder test failed: {e}")
        return False

def test_metrics():
    """测试评估指标"""
    print("\n=== Testing Metrics ===")
    
    try:
        from utils.metrics import MetricsCalculator
        
        # 创建计算器
        calculator = MetricsCalculator(device="cpu")
        
        # 创建测试图像
        img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # 测试PSNR和SSIM
        psnr_value = calculator.calculate_psnr(img1, img2)
        ssim_value = calculator.calculate_ssim(img1, img2)
        
        print(f"PSNR: {psnr_value:.2f}")
        print(f"SSIM: {ssim_value:.4f}")
        
        # 测试频域相似性
        freq_sim = calculator.calculate_frequency_similarity(img1, img2)
        print(f"Frequency similarity: {freq_sim:.4f}")
        
        print("✅ Metrics test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False

def run_full_test():
    """运行完整测试"""
    print("🚀 Starting Full Pipeline Test")
    print("=" * 50)
    
    # 创建测试数据
    create_test_data("./test_data", num_users=5, images_per_user=5)
    
    # 运行各项测试
    tests = [
        ("Data Loader", test_data_loader),
        ("VAE Model", test_vae_model),
        ("UNet Model", test_unet_model),
        ("Condition Encoder", test_condition_encoder),
        ("Metrics", test_metrics),
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # 总结结果
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Your environment is ready for training.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    # 清理测试数据
    if Path("./test_data").exists():
        shutil.rmtree("./test_data")
        print("\n🧹 Cleaned up test data")

def main():
    parser = argparse.ArgumentParser(description="Test Micro-Doppler Pipeline")
    parser.add_argument("--test", type=str, choices=[
        "all", "data", "vae", "unet", "encoder", "metrics"
    ], default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test == "all":
        run_full_test()
    elif args.test == "data":
        create_test_data("./test_data")
        test_data_loader()
    elif args.test == "vae":
        test_vae_model()
    elif args.test == "unet":
        test_unet_model()
    elif args.test == "encoder":
        test_condition_encoder()
    elif args.test == "metrics":
        test_metrics()

if __name__ == "__main__":
    main()
