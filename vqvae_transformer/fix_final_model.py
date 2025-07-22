#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复最终模型：用最佳模型权重替换可能过拟合的最终模型
"""

import torch
import shutil
from pathlib import Path
import argparse

def fix_final_model(vqvae_dir):
    """用best_model.pth权重替换final_model"""
    vqvae_path = Path(vqvae_dir)

    # 检查文件是否存在
    best_model_path = vqvae_path / "best_model.pth"
    final_model_path = vqvae_path / "final_model"

    if not best_model_path.exists():
        print(f"错误：未找到best_model.pth文件：{best_model_path}")
        return False

    if not final_model_path.exists():
        print(f"错误：未找到final_model目录：{final_model_path}")
        return False

    print(f"检查模型文件：")
    print(f"   best_model.pth: {best_model_path.stat().st_size / 1024 / 1024:.1f} MB")

    safetensors_path = final_model_path / "diffusion_pytorch_model.safetensors"
    if safetensors_path.exists():
        print(f"   final_model: {safetensors_path.stat().st_size / 1024 / 1024:.1f} MB")

    try:
        # 加载最佳模型权重
        print(f"\n正在加载最佳模型权重...")
        checkpoint = torch.load(best_model_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"成功加载最佳模型权重（轮次：{epoch}）")
        else:
            print(f"错误：检查点格式不正确")
            return False

        # 备份原始final_model
        backup_path = vqvae_path / "final_model_backup"
        if backup_path.exists():
            shutil.rmtree(backup_path)

        print(f"正在备份原始final_model...")
        shutil.copytree(final_model_path, backup_path)
        print(f"备份完成：{backup_path}")

        # 加载模型并用最佳权重更新
        print(f"正在用最佳权重更新final_model...")

        # 导入模型类
        import sys
        sys.path.append(str(vqvae_path.parent))
        from models.vqvae_model import MicroDopplerVQVAE

        # 从final_model加载模型结构
        model = MicroDopplerVQVAE.from_pretrained(final_model_path)

        # 用最佳权重更新
        model.load_state_dict(model_state)

        # 保存更新后的模型
        model.save_pretrained(final_model_path)

        print(f"成功用最佳权重更新final_model！")
        print(f"现在final_model使用第{epoch}轮的最佳权重")

        # 验证文件大小
        new_size = safetensors_path.stat().st_size / 1024 / 1024
        print(f"更新后文件大小：{new_size:.1f} MB")

        return True

    except Exception as e:
        print(f"更新失败：{e}")

        # 恢复备份
        if backup_path.exists():
            print(f"正在恢复原始final_model...")
            shutil.rmtree(final_model_path)
            shutil.move(backup_path, final_model_path)
            print(f"原始文件已恢复")

        return False

def main():
    parser = argparse.ArgumentParser(description="修复final_model以避免过拟合")
    parser.add_argument("--vqvae_dir", type=str,
                       default="/kaggle/working/outputs/vqvae_transformer/vqvae",
                       help="VQ-VAE输出目录")

    args = parser.parse_args()

    print("修复final_model工具")
    print("=" * 50)
    print("目标：用最佳模型权重替换可能过拟合的final_model")
    print(f"VQ-VAE目录：{args.vqvae_dir}")

    success = fix_final_model(args.vqvae_dir)

    if success:
        print("\n修复完成！")
        print("建议：")
        print("   1. 运行码本诊断以验证模型质量")
        print("   2. 开始Transformer训练")
        print("   3. 如果出现问题，从final_model_backup恢复")
    else:
        print("\n修复失败！")
        print("建议：")
        print("   1. 检查文件路径是否正确")
        print("   2. 确保best_model.pth和final_model都存在")
        print("   3. 手动使用best_model.pth进行后续训练")

if __name__ == "__main__":
    main()
