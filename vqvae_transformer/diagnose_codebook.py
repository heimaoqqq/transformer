#!/usr/bin/env python3
"""
码本诊断脚本
实时检查VQ-VAE训练过程中的码本状态
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_model_for_diagnosis(output_dir):
    """加载模型进行诊断 - 支持final_model和checkpoint"""
    output_path = Path(output_dir)

    # 优先尝试加载final_model (diffusers格式)
    final_model_path = output_path / "final_model"
    if final_model_path.exists():
        print(f"📂 检测到final_model目录: {final_model_path}")
        try:
            # 导入模型类
            import sys
            sys.path.append(str(output_path.parent.parent))
            from models.vqvae_model import MicroDopplerVQVAE

            # 加载模型
            model = MicroDopplerVQVAE.from_pretrained(final_model_path)
            print(f"✅ 成功加载final_model")

            # 创建伪checkpoint格式以兼容现有分析函数
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': 'final_model',
                'args': None,  # final_model中没有训练参数
            }
            return checkpoint

        except Exception as e:
            print(f"⚠️ final_model加载失败: {e}")
            print("🔄 尝试checkpoint格式...")

    # 备选：查找最新的checkpoint
    checkpoints = list(output_path.glob("checkpoint_epoch_*.pth"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        print(f"📂 加载checkpoint: {latest_checkpoint}")
        return torch.load(latest_checkpoint, map_location='cpu', weights_only=False)

    # 尝试best_model.pth
    best_model_path = output_path / "best_model.pth"
    if best_model_path.exists():
        print(f"📂 加载best_model: {best_model_path}")
        return torch.load(best_model_path, map_location='cpu', weights_only=False)

    print("❌ 未找到可用的模型文件")
    return None

def analyze_codebook_collapse(checkpoint):
    """分析码本坍缩情况"""
    print("\n🔍 码本坍缩诊断:")
    
    # 获取模型状态
    model_state = checkpoint['model_state_dict']
    
    # 查找量化器的嵌入权重
    embedding_key = None
    for key in model_state.keys():
        if 'quantize' in key and 'embedding' in key and 'weight' in key:
            embedding_key = key
            break
    
    if embedding_key is None:
        print("❌ 未找到量化器嵌入权重")
        return
    
    embeddings = model_state[embedding_key]  # [n_embed, embed_dim]
    print(f"📊 码本形状: {embeddings.shape}")
    
    # 计算嵌入向量之间的距离
    n_embed, embed_dim = embeddings.shape
    
    # 计算所有嵌入向量的成对距离
    distances = torch.cdist(embeddings, embeddings, p=2)
    
    # 去除对角线（自己与自己的距离）
    mask = ~torch.eye(n_embed, dtype=bool)
    min_distances = distances[mask].view(n_embed, n_embed-1).min(dim=1)[0]
    
    # 分析距离分布
    mean_min_dist = min_distances.mean().item()
    std_min_dist = min_distances.std().item()
    
    print(f"📏 最小距离统计:")
    print(f"   平均最小距离: {mean_min_dist:.4f}")
    print(f"   标准差: {std_min_dist:.4f}")
    
    # 检查坍缩指标
    collapse_threshold = 0.01  # 如果最小距离小于这个值，认为可能坍缩
    collapsed_codes = (min_distances < collapse_threshold).sum().item()
    
    print(f"🚨 坍缩分析:")
    print(f"   疑似坍缩码本数: {collapsed_codes}/{n_embed}")
    print(f"   坍缩比例: {collapsed_codes/n_embed*100:.1f}%")
    
    if collapsed_codes > n_embed * 0.1:  # 超过10%坍缩
        print("⚠️ 警告: 检测到严重码本坍缩!")
    elif collapsed_codes > 0:
        print("⚠️ 注意: 检测到轻微码本坍缩")
    else:
        print("✅ 码本状态良好，无明显坍缩")
    
    # 可视化距离分布
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(min_distances.numpy(), bins=50, alpha=0.7)
    plt.axvline(collapse_threshold, color='red', linestyle='--', label=f'坍缩阈值 ({collapse_threshold})')
    plt.xlabel('最小距离')
    plt.ylabel('频次')
    plt.title('码本最小距离分布')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.imshow(distances.numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('码本距离矩阵')
    plt.xlabel('码本索引')
    plt.ylabel('码本索引')
    
    plt.subplot(1, 3, 3)
    # 显示嵌入向量的范数
    norms = torch.norm(embeddings, dim=1)
    plt.hist(norms.numpy(), bins=50, alpha=0.7)
    plt.xlabel('嵌入向量范数')
    plt.ylabel('频次')
    plt.title('嵌入向量范数分布')
    
    plt.tight_layout()
    plt.savefig('codebook_diagnosis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'mean_min_distance': mean_min_dist,
        'collapsed_codes': collapsed_codes,
        'collapse_ratio': collapsed_codes/n_embed,
        'total_codes': n_embed
    }

def analyze_training_dynamics(checkpoint):
    """分析训练动态"""
    print("\n📈 训练动态分析:")

    epoch = checkpoint['epoch']
    print(f"📅 当前epoch: {epoch}")

    # 检查优化器状态
    if 'optimizer_state_dict' in checkpoint:
        optimizer_state = checkpoint['optimizer_state_dict']
        if 'param_groups' in optimizer_state:
            lr = optimizer_state['param_groups'][0]['lr']
            print(f"📚 当前学习率: {lr:.2e}")
    else:
        print("📚 学习率: N/A (final_model)")

    # 检查训练参数
    if 'args' in checkpoint and checkpoint['args'] is not None:
        args = checkpoint['args']
        print(f"🎯 训练配置:")
        print(f"   码本大小: {args.codebook_size}")
        print(f"   Commitment权重: {args.commitment_cost}")
        print(f"   EMA衰减: {getattr(args, 'ema_decay', 'N/A')}")
    else:
        print("🎯 训练配置: N/A (final_model格式)")

def main():
    parser = argparse.ArgumentParser(description="VQ-VAE码本诊断工具")
    parser.add_argument("--output_dir", type=str, 
                       default="/kaggle/working/outputs/vqvae_transformer/vqvae",
                       help="VQ-VAE输出目录")
    
    args = parser.parse_args()
    
    print("🔬 VQ-VAE码本诊断工具")
    print("=" * 50)
    
    # 加载模型进行诊断
    checkpoint = load_model_for_diagnosis(args.output_dir)
    if checkpoint is None:
        return
    
    # 分析训练动态
    analyze_training_dynamics(checkpoint)
    
    # 分析码本坍缩
    stats = analyze_codebook_collapse(checkpoint)
    
    print("\n💡 建议:")
    if stats['collapse_ratio'] > 0.1:
        print("🔧 严重坍缩，建议:")
        print("   1. 降低commitment_cost (如0.1)")
        print("   2. 增加ema_decay (如0.995)")
        print("   3. 降低学习率")
    elif stats['collapse_ratio'] > 0.05:
        print("⚠️ 轻微坍缩，建议:")
        print("   1. 监控后续训练")
        print("   2. 考虑调整commitment_cost")
    else:
        print("✅ 码本状态良好，继续训练")

if __name__ == "__main__":
    main()
