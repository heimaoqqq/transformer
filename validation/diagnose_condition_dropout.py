#!/usr/bin/env python3
"""
诊断条件dropout问题
检查条件扩散模型是否因为条件dropout而没有学到用户特征
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def analyze_condition_dropout_impact():
    """分析条件dropout对训练的影响"""
    print("🔍 分析条件dropout对训练的影响")
    
    # 从训练代码中我们看到：
    # if np.random.random() < args.condition_dropout:
    #     user_conditions = torch.zeros_like(user_indices)  # 无条件
    # else:
    #     user_conditions = user_indices  # 有条件
    
    print("\n📋 训练代码分析:")
    print("  条件dropout逻辑:")
    print("    - 如果随机数 < condition_dropout: 使用零向量(无条件)")
    print("    - 否则: 使用真实用户ID")
    
    # 模拟不同dropout率的影响
    dropout_rates = [0.0, 0.1, 0.2, 0.5, 0.9]
    
    print(f"\n📊 不同dropout率的影响分析:")
    for dropout_rate in dropout_rates:
        # 模拟1000次训练步骤
        num_steps = 1000
        conditional_steps = 0
        
        for _ in range(num_steps):
            if np.random.random() >= dropout_rate:  # 注意：这里是>=，表示使用条件
                conditional_steps += 1
        
        conditional_ratio = conditional_steps / num_steps
        
        print(f"  Dropout率 {dropout_rate:.1f}: 条件训练比例 {conditional_ratio:.1%}")
        
        if conditional_ratio < 0.5:
            print(f"    ❌ 条件训练不足，模型可能学不到用户特征")
        elif conditional_ratio < 0.8:
            print(f"    ⚠️  条件训练较少，用户特征可能较弱")
        else:
            print(f"    ✅ 条件训练充分")
    
    print(f"\n💡 关键发现:")
    print(f"  1. 条件dropout的目的是让模型学会无条件生成")
    print(f"  2. 但如果dropout率太高，模型主要学无条件生成")
    print(f"  3. 这会导致用户特征很弱或完全没有")
    
    return True

def test_condition_encoder_embeddings(condition_encoder_path: str, num_users: int = 31):
    """测试条件编码器的嵌入是否有意义"""
    print(f"\n🧠 测试条件编码器嵌入")
    
    if not Path(condition_encoder_path).exists():
        print(f"❌ 条件编码器文件不存在: {condition_encoder_path}")
        return False
    
    try:
        # 加载条件编码器
        from training.train_diffusion import UserConditionEncoder
        
        # 创建条件编码器 (需要知道embed_dim)
        condition_encoder = UserConditionEncoder(
            num_users=num_users,
            embed_dim=512  # 假设是512，与训练时一致
        )
        
        # 加载权重
        state_dict = torch.load(condition_encoder_path, map_location='cpu')
        condition_encoder.load_state_dict(state_dict)
        condition_encoder.eval()
        
        print(f"✅ 成功加载条件编码器")
        
        # 测试不同用户的嵌入
        with torch.no_grad():
            user_embeddings = []
            for user_idx in range(min(5, num_users)):  # 测试前5个用户
                user_tensor = torch.tensor([user_idx])
                embedding = condition_encoder(user_tensor)
                user_embeddings.append(embedding.numpy())
                
                print(f"  用户 {user_idx}: 嵌入范数 {torch.norm(embedding).item():.3f}")
        
        # 分析嵌入差异
        if len(user_embeddings) >= 2:
            user_embeddings = np.array(user_embeddings)
            
            # 计算用户间的余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(user_embeddings)
            
            print(f"\n📊 用户嵌入相似度分析:")
            avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            print(f"  平均相似度: {avg_similarity:.3f}")
            
            if avg_similarity > 0.95:
                print(f"  ❌ 用户嵌入几乎相同，没有学到用户特征")
                return False
            elif avg_similarity > 0.8:
                print(f"  ⚠️  用户嵌入相似度较高，用户特征较弱")
                return True
            else:
                print(f"  ✅ 用户嵌入有明显差异，学到了用户特征")
                return True
        
    except Exception as e:
        print(f"❌ 测试条件编码器失败: {e}")
        return False

def simulate_generation_with_different_conditions():
    """模拟不同条件下的生成效果"""
    print(f"\n🎨 模拟不同条件下的生成效果")
    
    # 模拟条件编码器
    embed_dim = 512
    num_users = 5
    
    # 情况1：用户嵌入几乎相同（条件dropout太高的结果）
    print(f"\n情况1: 用户嵌入几乎相同")
    similar_embeddings = []
    base_embedding = np.random.randn(embed_dim)
    for i in range(num_users):
        # 添加很小的噪声
        embedding = base_embedding + np.random.randn(embed_dim) * 0.01
        similar_embeddings.append(embedding)
    
    similar_embeddings = np.array(similar_embeddings)
    from sklearn.metrics.pairwise import cosine_similarity
    sim_similarities = cosine_similarity(similar_embeddings)
    avg_sim = np.mean(sim_similarities[np.triu_indices_from(sim_similarities, k=1)])
    print(f"  平均相似度: {avg_sim:.3f}")
    print(f"  预期效果: 所有用户生成的图像都很相似")
    
    # 情况2：用户嵌入有明显差异（正常训练的结果）
    print(f"\n情况2: 用户嵌入有明显差异")
    diverse_embeddings = []
    for i in range(num_users):
        # 每个用户有独特的嵌入
        embedding = np.random.randn(embed_dim)
        diverse_embeddings.append(embedding)
    
    diverse_embeddings = np.array(diverse_embeddings)
    div_similarities = cosine_similarity(diverse_embeddings)
    avg_div = np.mean(div_similarities[np.triu_indices_from(div_similarities, k=1)])
    print(f"  平均相似度: {avg_div:.3f}")
    print(f"  预期效果: 不同用户生成的图像有明显差异")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="诊断条件dropout问题")
    parser.add_argument("--condition_encoder_path", type=str, help="条件编码器路径")
    parser.add_argument("--num_users", type=int, default=31, help="用户数量")
    
    args = parser.parse_args()
    
    print("🔍 诊断条件dropout问题")
    print("=" * 50)
    
    # 分析条件dropout影响
    analyze_condition_dropout_impact()
    
    # 测试条件编码器嵌入
    if args.condition_encoder_path:
        test_condition_encoder_embeddings(args.condition_encoder_path, args.num_users)
    
    # 模拟生成效果
    simulate_generation_with_different_conditions()
    
    print("\n" + "=" * 50)
    print("🎯 诊断结论:")
    print("1. 如果条件dropout率太高(>0.5)，模型主要学无条件生成")
    print("2. 这会导致用户嵌入相似，生成图像缺乏用户特征")
    print("3. 建议检查训练时使用的condition_dropout参数")
    print("4. 如果用户嵌入相似度>0.9，说明没有学到用户特征")
    
    print("\n💡 解决方案:")
    print("1. 降低condition_dropout率到0.1或更低")
    print("2. 重新训练模型")
    print("3. 或者使用更强的指导强度来强化用户特征")

if __name__ == "__main__":
    main()
