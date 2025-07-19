#!/usr/bin/env python3
"""
训练用户验证分类器
为每个用户训练独立的ResNet-18二分类器
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from validation.user_classifier import UserValidationSystem

def find_user_data_directories(data_root: str) -> dict:
    """
    查找用户数据目录
    
    Args:
        data_root: 数据根目录
        
    Returns:
        用户ID到目录路径的映射
    """
    data_root = Path(data_root)
    user_dirs = {}
    
    print(f"🔍 在 {data_root} 中查找用户数据...")

    # 查找用户目录 (支持多种格式: user_01, user_1, ID_1, 1)
    for user_dir in data_root.iterdir():
        if user_dir.is_dir():
            dir_name = user_dir.name
            user_id = None

            try:
                if dir_name.startswith('user_'):
                    user_id = int(dir_name.split('_')[1])
                elif dir_name.startswith('ID_'):
                    user_id = int(dir_name.split('_')[1])
                elif dir_name.isdigit():
                    user_id = int(dir_name)

                if user_id is not None:
                    user_dirs[user_id] = str(user_dir)
                    print(f"  找到用户 {user_id}: {user_dir}")
                else:
                    print(f"  跳过无效目录: {user_dir}")

            except (IndexError, ValueError):
                print(f"  跳过无效目录: {user_dir}")
    
    print(f"✅ 找到 {len(user_dirs)} 个用户目录")
    return user_dirs

def train_classifiers_for_users(
    user_ids: List[int],
    real_data_root: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    max_samples_per_class: int = 500
):
    """
    为指定用户训练分类器
    
    Args:
        user_ids: 要训练的用户ID列表
        real_data_root: 真实数据根目录
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        max_samples_per_class: 每类最大样本数
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找用户数据目录
    user_dirs = find_user_data_directories(real_data_root)
    
    # 检查指定用户是否存在
    missing_users = [uid for uid in user_ids if uid not in user_dirs]
    if missing_users:
        print(f"❌ 以下用户数据不存在: {missing_users}")
        return
    
    # 初始化验证系统
    validation_system = UserValidationSystem()
    
    # 为每个用户训练分类器
    all_histories = {}
    
    for user_id in user_ids:
        print(f"\n{'='*50}")
        print(f"🎯 开始训练用户 {user_id} 的分类器")
        print(f"{'='*50}")
        
        # 准备该用户的数据
        user_real_dir = user_dirs[user_id]
        other_user_dirs = [user_dirs[uid] for uid in user_dirs.keys() if uid != user_id]
        
        # 准备训练数据
        image_paths, labels = validation_system.prepare_user_data(
            user_id=user_id,
            real_images_dir=user_real_dir,
            other_users_dirs=other_user_dirs,
            max_samples_per_class=max_samples_per_class
        )
        
        if len(image_paths) == 0:
            print(f"❌ 用户 {user_id} 没有可用的训练数据，跳过")
            continue
        
        # 检查数据平衡性
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        print(f"  数据分布: 正样本 {positive_count}, 负样本 {negative_count}")
        
        if positive_count < 10 or negative_count < 10:
            print(f"⚠️  用户 {user_id} 数据量过少，可能影响训练效果")
        
        # 训练分类器
        try:
            history = validation_system.train_user_classifier(
                user_id=user_id,
                image_paths=image_paths,
                labels=labels,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            all_histories[user_id] = history
            
            # 保存分类器
            classifier_path = output_path / f"user_{user_id:02d}_classifier.pth"
            validation_system.save_classifier(user_id, str(classifier_path))
            
            # 保存训练历史
            history_path = output_path / f"user_{user_id:02d}_history.json"
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            # 绘制训练曲线
            plot_path = output_path / f"user_{user_id:02d}_training.png"
            validation_system.plot_training_history(history, str(plot_path))
            
        except Exception as e:
            print(f"❌ 用户 {user_id} 训练失败: {e}")
            continue
    
    # 保存训练配置
    config = {
        'user_ids': user_ids,
        'real_data_root': real_data_root,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_samples_per_class': max_samples_per_class,
        'trained_users': list(all_histories.keys())
    }
    
    config_path = output_path / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n🎉 训练完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"✅ 成功训练 {len(all_histories)} 个用户分类器")

def main():
    parser = argparse.ArgumentParser(description="训练用户验证分类器")
    
    # 数据参数
    parser.add_argument("--real_data_root", type=str, required=True, 
                       help="真实数据根目录 (包含user_01, user_02等子目录)")
    parser.add_argument("--user_ids", type=int, nargs="+", required=True,
                       help="要训练的用户ID列表")
    parser.add_argument("--output_dir", type=str, default="./user_classifiers",
                       help="输出目录")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--max_samples_per_class", type=int, default=500,
                       help="每类最大样本数")
    
    args = parser.parse_args()
    
    print("🎯 用户验证分类器训练")
    print("=" * 50)
    print(f"真实数据根目录: {args.real_data_root}")
    print(f"用户ID列表: {args.user_ids}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    print("=" * 50)
    
    train_classifiers_for_users(
        user_ids=args.user_ids,
        real_data_root=args.real_data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples_per_class=args.max_samples_per_class
    )

if __name__ == "__main__":
    main()
