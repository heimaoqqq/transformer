#!/usr/bin/env python3
"""
使用diffusers标准组件训练Transformer的主入口
基于成熟的、经过验证的diffusers实现
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from training.train_diffusers_transformer import DiffusersTransformerTrainer

def main():
    parser = argparse.ArgumentParser(description="使用diffusers标准组件训练Transformer")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--vqvae_path", type=str, required=True, help="VQ-VAE模型路径")
    parser.add_argument("--output_dir", type=str, default="./diffusers_transformer_output", help="输出目录")
    
    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=1024, help="词汇表大小")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--num_users", type=int, default=31, help="用户数量")
    parser.add_argument("--num_layers", type=int, default=8, help="Transformer层数")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--attention_head_dim", type=int, default=64, help="注意力头维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--save_every", type=int, default=10, help="保存检查点间隔")
    
    args = parser.parse_args()
    
    print("🚀 使用diffusers标准组件训练Transformer")
    print("=" * 60)
    print(f"📂 数据目录: {args.data_dir}")
    print(f"🤖 VQ-VAE路径: {args.vqvae_path}")
    print(f"💾 输出目录: {args.output_dir}")
    print(f"📊 批次大小: {args.batch_size}")
    print(f"🔄 训练轮数: {args.num_epochs}")
    print(f"📈 学习率: {args.learning_rate}")
    print("=" * 60)
    
    # 创建训练器并开始训练
    trainer = DiffusersTransformerTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
