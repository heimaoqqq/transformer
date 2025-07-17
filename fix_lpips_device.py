#!/usr/bin/env python3
"""
修复LPIPS设备问题的临时脚本
如果LPIPS有设备问题，可以临时禁用感知损失
"""

import sys
from pathlib import Path

def disable_perceptual_loss():
    """临时禁用感知损失，避免设备问题"""
    print("🔧 临时禁用感知损失以避免设备问题...")
    
    # 修改train_celeba_standard.py
    train_script = Path("train_celeba_standard.py")
    if train_script.exists():
        content = train_script.read_text(encoding='utf-8')
        
        # 将感知损失权重设为0
        content = content.replace(
            '"--perceptual_weight", "1.0"',
            '"--perceptual_weight", "0.0"'
        )
        
        train_script.write_text(content, encoding='utf-8')
        print("✅ 已临时禁用感知损失")
        print("   感知损失权重: 1.0 → 0.0")
        print("   这样可以避免LPIPS设备问题")
        
        return True
    else:
        print("❌ 未找到train_celeba_standard.py")
        return False

def enable_perceptual_loss():
    """重新启用感知损失"""
    print("🔧 重新启用感知损失...")
    
    # 修改train_celeba_standard.py
    train_script = Path("train_celeba_standard.py")
    if train_script.exists():
        content = train_script.read_text(encoding='utf-8')
        
        # 将感知损失权重设为1.0
        content = content.replace(
            '"--perceptual_weight", "0.0"',
            '"--perceptual_weight", "1.0"'
        )
        
        train_script.write_text(content, encoding='utf-8')
        print("✅ 已重新启用感知损失")
        print("   感知损失权重: 0.0 → 1.0")
        
        return True
    else:
        print("❌ 未找到train_celeba_standard.py")
        return False

def main():
    """主函数"""
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        if action == "disable":
            disable_perceptual_loss()
        elif action == "enable":
            enable_perceptual_loss()
        else:
            print("用法: python fix_lpips_device.py [disable|enable]")
    else:
        print("🔧 LPIPS设备问题修复工具")
        print("=" * 40)
        print("如果遇到LPIPS设备错误，可以:")
        print("1. 临时禁用感知损失: python fix_lpips_device.py disable")
        print("2. 重新启用感知损失: python fix_lpips_device.py enable")
        print()
        print("建议:")
        print("- 先禁用感知损失完成训练")
        print("- 训练完成后可以重新启用进行微调")
        
        # 默认禁用
        disable_perceptual_loss()

if __name__ == "__main__":
    main()
