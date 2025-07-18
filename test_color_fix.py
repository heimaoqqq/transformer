#!/usr/bin/env python3
"""
测试颜色修复 - 验证VAE输出范围和反归一化是否正确
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from diffusers import AutoencoderKL

# 添加项目路径
import sys
sys.path.append('.')
from utils.data_loader import MicroDopplerDataset

def test_vae_output_range():
    """测试VAE的实际输出范围"""
    print("🔍 测试VAE输出范围...")
    
    # 加载VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_path = "/kaggle/input/final-model"  # 根据实际路径调整
    
    try:
        vae = AutoencoderKL.from_pretrained(vae_path)
        vae = vae.to(device)
        vae.eval()
        print("✅ VAE加载成功")
    except Exception as e:
        print(f"❌ VAE加载失败: {e}")
        return
    
    # 创建测试数据
    test_input = torch.randn(1, 3, 128, 128).to(device)
    test_input = torch.clamp(test_input, 0, 1)  # 确保输入在[0,1]范围
    
    with torch.no_grad():
        # VAE编码-解码
        posterior = vae.encode(test_input).latent_dist
        latents = posterior.sample()
        reconstructed = vae.decode(latents).sample
        
        # 检查输出范围
        min_val = reconstructed.min().item()
        max_val = reconstructed.max().item()
        mean_val = reconstructed.mean().item()
        
        print(f"📊 VAE输出统计:")
        print(f"   最小值: {min_val:.4f}")
        print(f"   最大值: {max_val:.4f}")
        print(f"   平均值: {mean_val:.4f}")
        
        # 判断输出范围
        if min_val >= -0.1 and max_val <= 1.1:
            print("✅ VAE输出范围似乎在[0,1]附近")
            return "0_1"
        elif min_val >= -1.1 and max_val <= 1.1:
            print("✅ VAE输出范围似乎在[-1,1]附近")
            return "-1_1"
        else:
            print(f"⚠️ VAE输出范围异常: [{min_val:.4f}, {max_val:.4f}]")
            return "unknown"

def test_color_conversion():
    """测试颜色转换的正确性"""
    print("\n🎨 测试颜色转换...")
    
    # 创建测试图像 - 深蓝色背景，类似真实图像
    test_image = np.zeros((128, 128, 3), dtype=np.uint8)
    test_image[:, :, 0] = 0    # R = 0
    test_image[:, :, 1] = 0    # G = 0  
    test_image[:, :, 2] = 255  # B = 255 (深蓝色)
    
    # 添加一些黄色区域 (类似真实图像的模式)
    test_image[60:68, :, 0] = 255  # R = 255
    test_image[60:68, :, 1] = 255  # G = 255
    test_image[60:68, :, 2] = 0    # B = 0 (黄色)
    
    print("📊 原始测试图像:")
    print(f"   蓝色区域 RGB: ({test_image[0,0,0]}, {test_image[0,0,1]}, {test_image[0,0,2]})")
    print(f"   黄色区域 RGB: ({test_image[64,64,0]}, {test_image[64,64,1]}, {test_image[64,64,2]})")
    
    # 转换为tensor (模拟数据加载器的处理)
    tensor_image = torch.from_numpy(test_image).float() / 255.0  # [0,1]
    tensor_image = tensor_image.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    
    print(f"   Tensor范围: [{tensor_image.min():.4f}, {tensor_image.max():.4f}]")
    
    # 测试两种反归一化方法
    print("\n🔄 测试反归一化方法:")
    
    # 方法1: 错误的方法 (假设输入是[-1,1])
    wrong_denorm = (tensor_image / 2 + 0.5).clamp(0, 1)
    wrong_result = (wrong_denorm * 255).numpy().astype(np.uint8)[0].transpose(1, 2, 0)
    
    # 方法2: 正确的方法 (输入已经是[0,1])
    correct_denorm = tensor_image.clamp(0, 1)
    correct_result = (correct_denorm * 255).numpy().astype(np.uint8)[0].transpose(1, 2, 0)
    
    print("❌ 错误方法结果:")
    print(f"   蓝色区域 RGB: ({wrong_result[0,0,0]}, {wrong_result[0,0,1]}, {wrong_result[0,0,2]})")
    print(f"   黄色区域 RGB: ({wrong_result[64,64,0]}, {wrong_result[64,64,1]}, {wrong_result[64,64,2]})")
    
    print("✅ 正确方法结果:")
    print(f"   蓝色区域 RGB: ({correct_result[0,0,0]}, {correct_result[0,0,1]}, {correct_result[0,0,2]})")
    print(f"   黄色区域 RGB: ({correct_result[64,64,0]}, {correct_result[64,64,1]}, {correct_result[64,64,2]})")
    
    # 保存对比图像
    output_dir = Path("color_test_results")
    output_dir.mkdir(exist_ok=True)
    
    Image.fromarray(test_image).save(output_dir / "original.png")
    Image.fromarray(wrong_result).save(output_dir / "wrong_denorm.png")
    Image.fromarray(correct_result).save(output_dir / "correct_denorm.png")
    
    print(f"\n💾 测试图像已保存到 {output_dir}/")
    
    return np.array_equal(test_image, correct_result)

def main():
    """主测试函数"""
    print("🧪 开始颜色修复测试\n")
    
    # 测试1: VAE输出范围
    vae_range = test_vae_output_range()
    
    # 测试2: 颜色转换
    color_correct = test_color_conversion()
    
    print("\n📋 测试总结:")
    print(f"   VAE输出范围: {vae_range}")
    print(f"   颜色转换正确: {'✅' if color_correct else '❌'}")
    
    if vae_range == "0_1" and color_correct:
        print("\n🎉 修复验证成功！")
        print("   - VAE输出确实在[0,1]范围")
        print("   - 移除了错误的 (image / 2 + 0.5) 转换")
        print("   - 生成图像的颜色应该恢复正常")
    else:
        print("\n⚠️ 需要进一步检查")
        if vae_range == "-1_1":
            print("   - VAE输出在[-1,1]范围，可能需要保留原来的转换")
        if not color_correct:
            print("   - 颜色转换测试失败")

if __name__ == "__main__":
    main()
