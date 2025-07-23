#!/usr/bin/env python3
"""
诊断工具使用说明和自动检测脚本
"""

import os
import sys
from pathlib import Path

def find_model_files():
    """自动查找模型文件"""
    print("🔍 自动查找模型文件...")
    
    # 查找VQ-VAE模型
    vqvae_paths = [
        "models/vqvae_model",
        "vqvae_transformer/models/vqvae_model",
        "models/vqvae",
        "vqvae_model"
    ]
    
    vqvae_path = None
    for path in vqvae_paths:
        if Path(path).exists():
            if Path(path + "/config.json").exists() or list(Path(path).glob("*.pth")):
                vqvae_path = path
                break
    
    # 查找Transformer模型
    transformer_paths = [
        "models/transformer_model",
        "models/transformer",
        "vqvae_transformer/models/transformer_model",
        "output/best_model*.pth",
        "models/transformer_improved"
    ]
    
    transformer_path = None
    for path in transformer_paths:
        if "*" in path:
            # 通配符搜索
            files = list(Path(".").glob(path))
            if files:
                transformer_path = str(files[0])
                break
        elif Path(path).exists():
            if Path(path).is_file() or list(Path(path).glob("*.pth")):
                transformer_path = path
                break
    
    # 查找数据目录
    data_paths = [
        "data/processed",
        "data",
        "vqvae_transformer/data/processed",
        "../data/processed"
    ]
    
    data_path = None
    for path in data_paths:
        if Path(path).exists():
            data_path = path
            break
    
    return vqvae_path, transformer_path, data_path

def print_usage():
    """打印使用说明"""
    print("🎯 遵循指南：组件诊断工具使用说明")
    print("="*60)
    print()
    
    print("📋 可用的诊断工具：")
    print()
    
    print("1️⃣ 快速检查 (推荐，5分钟内完成)")
    print("   python quick_component_check.py")
    print("   - 快速判断是VQ-VAE还是Transformer的问题")
    print("   - 提供明确的修复建议")
    print()
    
    print("2️⃣ 详细诊断 (完整分析，需要更多时间)")
    print("   python vqvae_transformer/diagnose_components.py")
    print("   - 深入分析每个组件的问题")
    print("   - 生成详细的诊断报告和图像")
    print()
    
    print("3️⃣ 自动诊断 (本脚本)")
    print("   python diagnose_usage.py --auto")
    print("   - 自动查找模型文件并运行诊断")
    print()
    
    print("📁 手动指定路径：")
    print("   --vqvae_path: VQ-VAE模型路径")
    print("   --transformer_path: Transformer模型路径")
    print("   --data_dir: 数据目录路径")
    print()
    
    print("💡 使用示例：")
    print("   # 快速检查（自动查找模型）")
    print("   python quick_component_check.py")
    print()
    print("   # 指定模型路径")
    print("   python quick_component_check.py \\")
    print("     --vqvae_path models/vqvae_model \\")
    print("     --transformer_path models/best_model.pth")
    print()

def auto_diagnose():
    """自动诊断"""
    print("🤖 自动诊断模式")
    print("="*40)
    
    # 查找模型文件
    vqvae_path, transformer_path, data_path = find_model_files()
    
    print(f"📁 发现的文件：")
    print(f"   VQ-VAE: {vqvae_path or '未找到'}")
    print(f"   Transformer: {transformer_path or '未找到'}")
    print(f"   数据目录: {data_path or '未找到'}")
    print()
    
    if not vqvae_path:
        print("❌ 未找到VQ-VAE模型，请先训练VQ-VAE")
        print("💡 运行: python vqvae_transformer/training/train_vqvae.py")
        return
    
    if not data_path:
        print("❌ 未找到数据目录，请检查数据路径")
        return
    
    # 构建命令
    cmd_parts = [
        sys.executable,
        "quick_component_check.py",
        f"--vqvae_path {vqvae_path}",
        f"--data_dir {data_path}"
    ]
    
    if transformer_path:
        cmd_parts.append(f"--transformer_path {transformer_path}")
    
    cmd = " ".join(cmd_parts)
    
    print(f"🚀 执行诊断命令：")
    print(f"   {cmd}")
    print()
    
    # 执行诊断
    os.system(cmd)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="诊断工具使用说明")
    parser.add_argument("--auto", action="store_true", help="自动查找模型并诊断")
    
    args = parser.parse_args()
    
    if args.auto:
        auto_diagnose()
    else:
        print_usage()
        
        # 询问是否运行自动诊断
        print("🤔 是否运行自动诊断？")
        print("   输入 'y' 或 'yes' 运行自动诊断")
        print("   输入其他任何内容退出")
        
        try:
            response = input("请选择: ").strip().lower()
            if response in ['y', 'yes']:
                print()
                auto_diagnose()
        except KeyboardInterrupt:
            print("\n👋 退出")

if __name__ == "__main__":
    main()
