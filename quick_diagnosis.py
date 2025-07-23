#!/usr/bin/env python3
"""
快速诊断脚本：检查PSNR停滞问题
分析生成的token序列和图像质量
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def analyze_generated_tokens():
    """分析生成的token序列"""
    print("🔍 分析生成的token序列...")
    
    # 模拟生成的token序列（从训练日志可以看到范围是[0, 1019]）
    # 这里我们分析几种可能的情况
    
    # 情况1：完全随机的token
    random_tokens = torch.randint(0, 1020, (5, 1024))
    print(f"随机token示例:")
    print(f"  唯一token数量: {torch.unique(random_tokens[0]).shape[0]}/1024")
    print(f"  token分布: min={random_tokens.min()}, max={random_tokens.max()}")
    
    # 情况2：重复的token（模式崩溃）
    repeated_tokens = torch.full((5, 1024), 100)  # 全是100
    print(f"\n重复token示例:")
    print(f"  唯一token数量: {torch.unique(repeated_tokens[0]).shape[0]}/1024")
    
    # 情况3：有限多样性的token
    limited_tokens = torch.randint(0, 50, (5, 1024))  # 只使用前50个token
    print(f"\n有限多样性token示例:")
    print(f"  唯一token数量: {torch.unique(limited_tokens[0]).shape[0]}/1024")
    print(f"  token分布: min={limited_tokens.min()}, max={limited_tokens.max()}")

def check_vqvae_decoding():
    """检查VQ-VAE解码是否正常"""
    print("\n🔍 检查VQ-VAE解码...")
    
    try:
        from models.vqvae_model import MicroDopplerVQVAE
        
        # 这里需要实际的VQ-VAE模型路径
        print("需要加载实际的VQ-VAE模型来测试解码...")
        print("建议手动运行以下测试:")
        print("1. 加载VQ-VAE模型")
        print("2. 创建测试token序列")
        print("3. 测试解码过程")
        print("4. 检查输出图像是否合理")
        
    except Exception as e:
        print(f"无法加载VQ-VAE模型: {e}")

def analyze_psnr_calculation():
    """分析PSNR计算是否正确"""
    print("\n🔍 分析PSNR计算...")
    
    # 模拟PSNR计算
    def calculate_psnr(img1, img2, max_val=2.0):
        """计算PSNR"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr.item()
    
    # 测试不同情况的PSNR
    # 情况1：完全相同的图像
    img1 = torch.randn(3, 128, 128)
    img2 = img1.clone()
    psnr_identical = calculate_psnr(img1, img2)
    print(f"相同图像PSNR: {psnr_identical:.2f} dB (应该是inf)")
    
    # 情况2：轻微差异的图像
    img2_slight = img1 + torch.randn_like(img1) * 0.01
    psnr_slight = calculate_psnr(img1, img2_slight)
    print(f"轻微差异PSNR: {psnr_slight:.2f} dB (应该很高)")
    
    # 情况3：很大差异的图像
    img2_large = torch.randn_like(img1)
    psnr_large = calculate_psnr(img1, img2_large)
    print(f"很大差异PSNR: {psnr_large:.2f} dB (应该很低)")
    
    # 情况4：黑色图像 vs 正常图像
    black_img = torch.zeros_like(img1)
    normal_img = torch.randn_like(img1) * 0.5
    psnr_black = calculate_psnr(black_img, normal_img)
    print(f"黑色vs正常图像PSNR: {psnr_black:.2f} dB")
    
    print(f"\n我们的训练PSNR: ~9 dB")
    print(f"这表明生成的图像与原图差异很大，可能接近随机噪声水平")

def suggest_debugging_steps():
    """建议调试步骤"""
    print("\n🎯 建议的调试步骤:")
    
    print("\n1. 检查生成的样本图像:")
    print("   - 查看 /kaggle/working/outputs/vqvae_transformer/transformer/samples/")
    print("   - 检查图像是否是黑色、噪声或有意义的结构")
    
    print("\n2. 分析token生成质量:")
    print("   - 检查生成的token是否有多样性")
    print("   - 验证不同用户是否生成不同的token序列")
    
    print("\n3. 验证VQ-VAE解码:")
    print("   - 用已知的好token测试VQ-VAE解码")
    print("   - 确认force_not_quantize=True是否正确")
    
    print("\n4. 检查评估逻辑:")
    print("   - 验证PSNR计算是否使用了正确的图像范围")
    print("   - 确认原图和生成图的预处理是否一致")
    
    print("\n5. 可能的修复方案:")
    print("   - 如果图像是黑色：VQ-VAE解码问题")
    print("   - 如果图像是噪声：生成模式崩溃")
    print("   - 如果图像看起来正常但PSNR低：评估逻辑问题")

def main():
    """主函数"""
    print("🚨 Transformer训练PSNR停滞问题诊断")
    print("="*50)
    
    analyze_generated_tokens()
    check_vqvae_decoding()
    analyze_psnr_calculation()
    suggest_debugging_steps()
    
    print("\n" + "="*50)
    print("📋 诊断完成！请根据建议进行进一步调试。")

if __name__ == "__main__":
    main()
