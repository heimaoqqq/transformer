#!/usr/bin/env python3
"""
LPIPS感知损失管理工具
- 安装LPIPS库
- 管理感知损失开关
- 解决设备兼容性问题
"""

import subprocess
import sys
from pathlib import Path

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss_fn = lpips.LPIPS(net='vgg').to(device)
        test_img = torch.randn(1, 3, 64, 64).to(device)
        with torch.no_grad():
            loss = loss_fn(test_img, test_img)
        print(f"✅ LPIPS功能测试成功: {loss.item():.6f} (设备: {device})")

        return True

    except Exception as e:
        print(f"❌ LPIPS安装失败: {e}")
        return False

def enable_perceptual_loss():
    """启用感知损失"""
    print("🔧 启用感知损失...")

    train_script = Path("train_celeba_standard.py")
    if train_script.exists():
        content = train_script.read_text(encoding='utf-8')

        # 将感知损失权重设为1.0
        if '"--perceptual_weight", "0.0"' in content:
            content = content.replace(
                '"--perceptual_weight", "0.0"',
                '"--perceptual_weight", "1.0"'
            )
            train_script.write_text(content, encoding='utf-8')
            print("✅ 已启用感知损失 (权重: 0.0 → 1.0)")
            return True
        elif '"--perceptual_weight", "1.0"' in content:
            print("✅ 感知损失已经启用 (权重: 1.0)")
            return True
        else:
            print("⚠️  未找到感知损失配置")
            return False
    else:
        print("❌ 未找到train_celeba_standard.py")
        return False

def disable_perceptual_loss():
    """禁用感知损失"""
    print("🔧 禁用感知损失...")

    train_script = Path("train_celeba_standard.py")
    if train_script.exists():
        content = train_script.read_text(encoding='utf-8')

        # 将感知损失权重设为0.0
        if '"--perceptual_weight", "1.0"' in content:
            content = content.replace(
                '"--perceptual_weight", "1.0"',
                '"--perceptual_weight", "0.0"'
            )
            train_script.write_text(content, encoding='utf-8')
            print("✅ 已禁用感知损失 (权重: 1.0 → 0.0)")
            return True
        elif '"--perceptual_weight", "0.0"' in content:
            print("✅ 感知损失已经禁用 (权重: 0.0)")
            return True
        else:
            print("⚠️  未找到感知损失配置")
            return False
    else:
        print("❌ 未找到train_celeba_standard.py")
        return False

def main():
    """主函数"""
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        if action == "install":
            success = install_lpips()
            if success:
                enable_perceptual_loss()
        elif action == "enable":
            enable_perceptual_loss()
        elif action == "disable":
            disable_perceptual_loss()
        elif action == "test":
            # 测试LPIPS是否可用
            try:
                import lpips
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _ = lpips.LPIPS(net='vgg').to(device)
                print(f"✅ LPIPS可用 (设备: {device})")
            except Exception as e:
                print(f"❌ LPIPS不可用: {e}")
        else:
            print("用法: python install_lpips.py [install|enable|disable|test]")
    else:
        # 默认行为：安装LPIPS并启用感知损失
        print("🚀 LPIPS感知损失管理工具")
        print("=" * 50)

        # 1. 安装LPIPS
        success = install_lpips()

        if success:
            # 2. 启用感知损失
            enable_perceptual_loss()

            print("\n🎉 设置完成！")
            print("✅ LPIPS已安装并测试通过")
            print("✅ 感知损失已启用 (权重: 1.0)")
            print("🚀 现在可以运行高质量VAE训练:")
            print("   python train_celeba_standard.py")
        else:
            # 3. 如果安装失败，禁用感知损失
            print("\n⚠️  LPIPS安装失败，禁用感知损失")
            disable_perceptual_loss()
            print("🔄 可以使用MSE损失进行训练:")
            print("   python train_celeba_standard.py")

        print("\n📝 其他选项:")
        print("   python install_lpips.py enable   # 启用感知损失")
        print("   python install_lpips.py disable  # 禁用感知损失")
        print("   python install_lpips.py test     # 测试LPIPS可用性")

if __name__ == "__main__":
    main()
