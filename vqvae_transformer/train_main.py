#!/usr/bin/env python3
"""
VQ-VAE + Transformer 主训练脚本
两阶段训练：
1. 训练VQ-VAE学习图像的离散表示
2. 训练Transformer从用户ID生成token序列
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import torch

def setup_environment():
    """设置训练环境"""
    # GPU优化
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"🎮 GPU: {torch.cuda.get_device_properties(0).name}")
        print(f"   内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️ 未检测到GPU，将使用CPU训练")

def get_optimized_config():
    """获取针对微多普勒优化的配置"""
    if not torch.cuda.is_available():
        return None
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if gpu_memory >= 14:  # P100, V100等
        return {
            "vqvae_batch_size": 16,
            "transformer_batch_size": 8,
            "num_workers": 4,
            "mixed_precision": False,  # P100不支持混合精度
        }
    elif gpu_memory >= 10:  # T4等
        return {
            "vqvae_batch_size": 12,
            "transformer_batch_size": 6,
            "num_workers": 2,
            "mixed_precision": True,
        }
    else:  # 低端GPU
        return {
            "vqvae_batch_size": 8,
            "transformer_batch_size": 4,
            "num_workers": 1,
            "mixed_precision": True,
        }

def train_vqvae(args, config):
    """训练VQ-VAE"""
    print("\n🎯 阶段1: 训练VQ-VAE")
    print("=" * 50)
    
    vqvae_output = Path(args.output_dir) / "vqvae"
    
    cmd = [
        "python", "training/train_vqvae.py",
        "--data_dir", args.data_dir,
        "--output_dir", str(vqvae_output),
        "--resolution", str(args.resolution),
        "--codebook_size", str(args.codebook_size),
        "--commitment_cost", str(args.commitment_cost),
        "--ema_decay", str(args.ema_decay),
        "--batch_size", str(config["vqvae_batch_size"]),
        "--num_epochs", str(args.vqvae_epochs),
        "--learning_rate", str(args.vqvae_lr),
        "--num_workers", str(config["num_workers"]),
        "--sample_interval", "500",
        "--eval_interval", "5",
        "--codebook_monitor_interval", "1",
        "--keep_checkpoints", "3",  # 只保留最近3个checkpoint
        "--milestone_interval", "10",  # 每10个epoch保存里程碑
        "--auto_cleanup",  # 启用自动清理
    ]
    
    print(f"🚀 启动VQ-VAE训练...")
    print(f"   命令: {' '.join(cmd)}")
    
    # 设置工作目录为vqvae_transformer
    cwd = Path(__file__).parent
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=cwd)
    
    if result.returncode != 0:
        print(f"❌ VQ-VAE训练失败")
        return False
    
    print(f"✅ VQ-VAE训练完成")
    return True

def train_transformer(args, config):
    """训练Transformer"""
    print("\n🎯 阶段2: 训练Transformer")
    print("=" * 50)

    # 确定VQ-VAE路径
    if args.vqvae_path:
        vqvae_path = Path(args.vqvae_path)
        print(f"📂 使用指定的VQ-VAE路径: {vqvae_path}")
    else:
        vqvae_path = Path(args.output_dir) / "vqvae"
        print(f"📂 使用默认VQ-VAE路径: {vqvae_path}")

    transformer_output = Path(args.output_dir) / "transformer"
    
    # 检查VQ-VAE是否存在
    final_model_exists = (vqvae_path / "final_model").exists()
    checkpoint_exists = (vqvae_path / "best_model.pth").exists() or len(list(vqvae_path.glob("*.pth"))) > 0

    if not final_model_exists and not checkpoint_exists:
        print(f"❌ 未找到VQ-VAE模型: {vqvae_path}")
        print(f"   期望文件: final_model/ 或 *.pth")
        return False

    if final_model_exists:
        print(f"✅ 找到VQ-VAE模型 (diffusers格式): {vqvae_path}/final_model")
    else:
        print(f"✅ 找到VQ-VAE模型 (checkpoint格式): {vqvae_path}/*.pth")
    
    cmd = [
        "python", "training/train_transformer.py",
        "--data_dir", args.data_dir,
        "--vqvae_path", str(vqvae_path),
        "--output_dir", str(transformer_output),
        "--resolution", str(args.resolution),
        "--codebook_size", str(args.codebook_size),
        "--num_users", str(args.num_users),
        "--n_embd", str(args.n_embd),
        "--n_layer", str(args.n_layer),
        "--n_head", str(args.n_head),
        "--batch_size", str(config["transformer_batch_size"]),
        "--num_epochs", str(args.transformer_epochs),
        "--learning_rate", str(args.transformer_lr),
        "--num_workers", str(config["num_workers"]),
        "--save_interval", "10",
        "--sample_interval", "10",
        "--generation_temperature", str(args.generation_temperature),
    ]
    
    if args.use_cross_attention:
        cmd.append("--use_cross_attention")
    
    print(f"🚀 启动Transformer训练...")
    print(f"   命令: {' '.join(cmd)}")
    
    # 设置工作目录为vqvae_transformer
    cwd = Path(__file__).parent
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=cwd)
    
    if result.returncode != 0:
        print(f"❌ Transformer训练失败")
        return False
    
    print(f"✅ Transformer训练完成")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VQ-VAE + Transformer 微多普勒生成系统")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/dataset",
                       help="数据集目录 (包含ID1, ID_2, ..., ID_31目录)")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/outputs/vqvae_transformer",
                       help="输出目录")
    parser.add_argument("--resolution", type=int, default=128,
                       help="图像分辨率")
    
    # VQ-VAE参数
    parser.add_argument("--vqvae_path", type=str, default=None,
                       help="预训练VQ-VAE模型路径 (如果不指定，使用output_dir/vqvae)")
    parser.add_argument("--codebook_size", type=int, default=1024,
                       help="码本大小")
    parser.add_argument("--commitment_cost", type=float, default=0.25,
                       help="Commitment损失权重")
    parser.add_argument("--ema_decay", type=float, default=0.99,
                       help="EMA衰减率")
    parser.add_argument("--vqvae_epochs", type=int, default=80,
                       help="VQ-VAE训练轮数")
    parser.add_argument("--vqvae_lr", type=float, default=1e-4,
                       help="VQ-VAE学习率")
    
    # Transformer参数
    parser.add_argument("--num_users", type=int, default=31,
                       help="用户数量")
    parser.add_argument("--n_embd", type=int, default=512,
                       help="Transformer嵌入维度")
    parser.add_argument("--n_layer", type=int, default=8,
                       help="Transformer层数")
    parser.add_argument("--n_head", type=int, default=8,
                       help="注意力头数")
    parser.add_argument("--transformer_epochs", type=int, default=50,
                       help="Transformer训练轮数")
    parser.add_argument("--transformer_lr", type=float, default=1e-4,
                       help="Transformer学习率")
    parser.add_argument("--use_cross_attention", action="store_true",
                       help="使用交叉注意力")
    
    # 生成参数
    parser.add_argument("--generation_temperature", type=float, default=1.0,
                       help="生成温度")
    
    # 训练控制
    parser.add_argument("--skip_vqvae", action="store_true",
                       help="跳过VQ-VAE训练")
    parser.add_argument("--skip_transformer", action="store_true",
                       help="跳过Transformer训练")
    
    args = parser.parse_args()
    
    print("🎨 VQ-VAE + Transformer 微多普勒生成系统")
    print("=" * 60)
    print(f"📊 数据集: {args.data_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🖼️ 分辨率: {args.resolution}x{args.resolution}")
    print(f"📚 码本大小: {args.codebook_size}")
    print(f"👥 用户数量: {args.num_users}")
    print(f"🧠 Transformer: {args.n_layer}层, {args.n_embd}维, {args.n_head}头")
    
    # 设置环境
    setup_environment()
    
    # 获取优化配置
    config = get_optimized_config()
    if not config:
        print("❌ GPU配置获取失败")
        return
    
    print(f"\n🔧 优化配置:")
    print(f"   VQ-VAE批次大小: {config['vqvae_batch_size']}")
    print(f"   Transformer批次大小: {config['transformer_batch_size']}")
    print(f"   工作进程数: {config['num_workers']}")
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    success = True
    
    # 阶段1: 训练VQ-VAE
    if not args.skip_vqvae:
        success = train_vqvae(args, config)
        if not success:
            print("❌ VQ-VAE训练失败，停止训练")
            return
    else:
        print("⏭️ 跳过VQ-VAE训练")
    
    # 阶段2: 训练Transformer
    if not args.skip_transformer and success:
        success = train_transformer(args, config)
        if not success:
            print("❌ Transformer训练失败")
            return
    else:
        print("⏭️ 跳过Transformer训练")
    
    if success:
        print("\n🎉 VQ-VAE + Transformer训练完成!")
        print(f"📁 模型保存在: {args.output_dir}")
        print(f"🔍 下一步: 运行推理和验证")
        print(f"   python generate_main.py --model_dir {args.output_dir}")
        print(f"   python validate_main.py --model_dir {args.output_dir}")
    else:
        print("\n❌ 训练过程中出现错误")

if __name__ == "__main__":
    main()
