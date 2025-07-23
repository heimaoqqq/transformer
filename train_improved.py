#!/usr/bin/env python3
"""
改进的Transformer训练脚本 - 解决生成模式崩溃问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vqvae_transformer.training.train_transformer import main
import argparse

def create_improved_args():
    """创建改进的训练参数"""
    parser = argparse.ArgumentParser(description="改进的Transformer训练")
    
    # 基础参数
    parser.add_argument("--data_dir", type=str, default="data/processed", help="数据目录")
    parser.add_argument("--vqvae_path", type=str, default="models/vqvae_model", help="VQ-VAE模型路径")
    parser.add_argument("--output_dir", type=str, default="models/transformer_improved", help="输出目录")
    
    # 模型参数 - 更保守的设置
    parser.add_argument("--resolution", type=int, default=128, help="图像分辨率")
    parser.add_argument("--codebook_size", type=int, default=1024, help="码本大小")
    parser.add_argument("--num_users", type=int, default=31, help="用户数量")
    
    # Transformer参数 - 减小模型复杂度
    parser.add_argument("--d_model", type=int, default=256, help="模型维度")  # 从512减少到256
    parser.add_argument("--nhead", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=6, help="层数")  # 从12减少到6
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="前馈网络维度")  # 从2048减少到1024
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout率")  # 增加正则化
    
    # 训练参数 - 更保守的学习
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")  # 减小批次
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")  # 降低学习率
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")  # 增加训练轮数
    parser.add_argument("--warmup_steps", type=int, default=2000, help="预热步数")  # 增加预热
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    
    # 生成参数 - 更保守的生成
    parser.add_argument("--generation_temperature", type=float, default=0.7, help="生成温度")  # 降低温度
    parser.add_argument("--max_seq_len", type=int, default=1024, help="最大序列长度")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--save_every", type=int, default=5, help="每N个epoch保存一次")
    parser.add_argument("--eval_every", type=int, default=2, help="每N个epoch评估一次")
    
    return parser.parse_args()

if __name__ == "__main__":
    print("🎯 遵循指南：启动改进的Transformer训练")
    print("=" * 60)
    print("🔧 改进措施:")
    print("   ✅ 添加空间一致性损失")
    print("   ✅ 使用Top-k采样策略")
    print("   ✅ 降低生成温度")
    print("   ✅ 减小模型复杂度")
    print("   ✅ 降低学习率")
    print("   ✅ 增加正则化")
    print("   ✅ VQ-VAE质量检查")
    print("=" * 60)
    
    # 创建改进的参数
    args = create_improved_args()
    
    # 显示关键参数
    print(f"📊 关键参数:")
    print(f"   模型维度: {args.d_model}")
    print(f"   层数: {args.num_layers}")
    print(f"   学习率: {args.learning_rate}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   生成温度: {args.generation_temperature}")
    print(f"   Dropout: {args.dropout}")
    print()
    
    # 替换sys.argv以传递参数
    sys.argv = ['train_transformer.py']
    for key, value in vars(args).items():
        sys.argv.extend([f'--{key}', str(value)])
    
    # 启动训练
    main()
