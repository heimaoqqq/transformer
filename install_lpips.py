#!/usr/bin/env python3
"""
安装LPIPS感知损失库
用于VAE训练的感知损失计算
"""

import subprocess
import sys

def install_lpips():
    """安装LPIPS库"""
    print("🔧 安装LPIPS感知损失库...")
    
    try:
        # 安装lpips
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lpips==0.1.4"])
        print("✅ LPIPS安装成功")
        
        # 测试导入
        import lpips
        print("✅ LPIPS导入测试成功")
        
        # 测试功能
        import torch
        loss_fn = lpips.LPIPS(net='vgg')
        test_img = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            loss = loss_fn(test_img, test_img)
        print(f"✅ LPIPS功能测试成功: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ LPIPS安装失败: {e}")
        return False

if __name__ == "__main__":
    success = install_lpips()
    if success:
        print("\n🎉 LPIPS安装完成！现在可以使用感知损失训练VAE")
    else:
        print("\n❌ LPIPS安装失败，将使用MSE损失训练")
