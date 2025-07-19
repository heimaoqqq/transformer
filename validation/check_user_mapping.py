#!/usr/bin/env python3
"""
用户ID映射一致性检查工具
确保训练、推理、验证时使用相同的用户ID映射
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def get_training_mapping(data_dir: str) -> Dict[int, int]:
    """获取训练时的用户ID映射（模拟data_loader.py的逻辑）"""
    print("🏋️  获取训练时的用户ID映射...")
    
    data_path = Path(data_dir)
    users = []
    
    for user_dir in data_path.iterdir():
        if user_dir.is_dir() and user_dir.name.startswith('ID_'):
            try:
                user_id = int(user_dir.name.split('_')[1])
                users.append(user_id)
            except ValueError:
                continue
    
    users = sorted(users)  # 训练时的排序逻辑
    user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
    
    print(f"  找到用户: {users}")
    print(f"  映射: {user_to_idx}")
    
    return user_to_idx

def get_inference_mapping(data_dir: str) -> Dict[int, int]:
    """获取推理时的用户ID映射（模拟generate_training_style.py的逻辑）"""
    print("\n🎨 获取推理时的用户ID映射...")
    
    data_path = Path(data_dir)
    all_users = []
    
    for user_dir in data_path.iterdir():
        if user_dir.is_dir() and user_dir.name.startswith('ID_'):
            try:
                user_id = int(user_dir.name.split('_')[1])
                all_users.append(user_id)
            except ValueError:
                continue
    
    all_users = sorted(all_users)  # 推理时的排序逻辑
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(all_users)}
    
    print(f"  找到用户: {all_users}")
    print(f"  映射: {user_id_to_idx}")
    
    return user_id_to_idx

def get_validation_mapping(data_dir: str) -> Dict[int, int]:
    """获取验证时的用户ID映射（模拟validation_pipeline.py的逻辑）"""
    print("\n🔍 获取验证时的用户ID映射...")
    
    data_path = Path(data_dir)
    all_users = []
    
    for user_dir in data_path.iterdir():
        if user_dir.is_dir() and user_dir.name.startswith('ID_'):
            try:
                user_id = int(user_dir.name.split('_')[1])
                all_users.append(user_id)
            except ValueError:
                continue
    
    all_users = sorted(all_users)  # 验证时的排序逻辑
    user_id_mapping = {user_id: idx for idx, user_id in enumerate(all_users)}
    
    print(f"  找到用户: {all_users}")
    print(f"  映射: {user_id_mapping}")
    
    return user_id_mapping

def check_mapping_consistency(data_dir: str) -> bool:
    """检查所有映射是否一致"""
    print("🔍 检查用户ID映射一致性")
    print("=" * 60)
    
    # 获取各个阶段的映射
    training_mapping = get_training_mapping(data_dir)
    inference_mapping = get_inference_mapping(data_dir)
    validation_mapping = get_validation_mapping(data_dir)
    
    # 检查一致性
    print(f"\n📊 一致性检查:")
    
    # 检查用户列表是否相同
    training_users = set(training_mapping.keys())
    inference_users = set(inference_mapping.keys())
    validation_users = set(validation_mapping.keys())
    
    if training_users == inference_users == validation_users:
        print(f"  ✅ 用户列表一致: {sorted(training_users)}")
    else:
        print(f"  ❌ 用户列表不一致!")
        print(f"    训练: {sorted(training_users)}")
        print(f"    推理: {sorted(inference_users)}")
        print(f"    验证: {sorted(validation_users)}")
        return False
    
    # 检查映射是否相同
    if training_mapping == inference_mapping == validation_mapping:
        print(f"  ✅ 用户ID映射完全一致")
        
        # 显示映射详情
        print(f"\n📋 统一的用户ID映射:")
        for user_id in sorted(training_mapping.keys()):
            idx = training_mapping[user_id]
            print(f"    用户 {user_id:2d} → 索引 {idx:2d}")
        
        return True
    else:
        print(f"  ❌ 用户ID映射不一致!")
        
        # 找出不一致的地方
        all_users = sorted(training_users)
        print(f"\n🔍 详细对比:")
        print(f"{'用户ID':<8} {'训练':<8} {'推理':<8} {'验证':<8} {'状态'}")
        print("-" * 50)
        
        for user_id in all_users:
            train_idx = training_mapping.get(user_id, "N/A")
            infer_idx = inference_mapping.get(user_id, "N/A")
            valid_idx = validation_mapping.get(user_id, "N/A")
            
            if train_idx == infer_idx == valid_idx:
                status = "✅"
            else:
                status = "❌"
            
            print(f"{user_id:<8} {train_idx:<8} {infer_idx:<8} {valid_idx:<8} {status}")
        
        return False

def check_data_directory_structure(data_dir: str):
    """检查数据目录结构"""
    print(f"\n📁 检查数据目录结构: {data_dir}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"  ❌ 数据目录不存在: {data_path}")
        return
    
    # 列出所有子目录
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"  📂 找到 {len(subdirs)} 个子目录:")
    
    user_dirs = []
    other_dirs = []
    
    for subdir in subdirs:
        if subdir.name.startswith('ID_'):
            try:
                user_id = int(subdir.name.split('_')[1])
                user_dirs.append((user_id, subdir))
            except ValueError:
                other_dirs.append(subdir)
        else:
            other_dirs.append(subdir)
    
    # 显示用户目录
    user_dirs.sort(key=lambda x: x[0])
    print(f"  👥 用户目录 ({len(user_dirs)} 个):")
    for user_id, user_dir in user_dirs:
        # 统计图像数量
        image_files = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))
        print(f"    ID_{user_id:2d} → {len(image_files):3d} 张图像")
    
    # 显示其他目录
    if other_dirs:
        print(f"  📁 其他目录 ({len(other_dirs)} 个):")
        for other_dir in other_dirs[:5]:  # 只显示前5个
            print(f"    {other_dir.name}")
        if len(other_dirs) > 5:
            print(f"    ... 还有 {len(other_dirs) - 5} 个")

def test_specific_user_mapping(data_dir: str, test_user_ids: List[int]):
    """测试特定用户的映射"""
    print(f"\n🎯 测试特定用户映射: {test_user_ids}")
    
    # 获取映射
    mapping = get_training_mapping(data_dir)
    
    print(f"\n📊 测试结果:")
    for user_id in test_user_ids:
        if user_id in mapping:
            idx = mapping[user_id]
            print(f"  用户 {user_id:2d} → 索引 {idx:2d} ✅")
        else:
            print(f"  用户 {user_id:2d} → 未找到 ❌")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="检查用户ID映射一致性")
    parser.add_argument("--data_dir", type=str, required=True, help="数据根目录")
    parser.add_argument("--test_users", type=int, nargs='+', default=[1, 2, 3],
                       help="测试特定用户ID")
    
    args = parser.parse_args()
    
    print("🔍 用户ID映射一致性检查工具")
    print("=" * 60)
    
    # 检查数据目录结构
    check_data_directory_structure(args.data_dir)
    
    # 检查映射一致性
    is_consistent = check_mapping_consistency(args.data_dir)
    
    # 测试特定用户
    test_specific_user_mapping(args.data_dir, args.test_users)
    
    # 总结
    print(f"\n" + "=" * 60)
    if is_consistent:
        print("🎉 用户ID映射完全一致，没有问题！")
        print("💡 如果验证仍然失败，问题可能在其他地方")
    else:
        print("🚨 发现用户ID映射不一致！")
        print("💡 这很可能是验证失败的根本原因")
        print("🔧 建议检查并统一所有组件的映射逻辑")

if __name__ == "__main__":
    main()
