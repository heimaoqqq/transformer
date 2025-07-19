#!/usr/bin/env python3
"""
验证生成图像
使用训练好的用户分类器验证生成图像是否包含对应用户特征
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from validation.user_classifier import UserValidationSystem

def find_generated_images(generated_root: str, user_ids: List[int]) -> Dict[int, str]:
    """
    查找生成图像目录
    
    Args:
        generated_root: 生成图像根目录
        user_ids: 用户ID列表
        
    Returns:
        用户ID到生成图像目录的映射
    """
    generated_root = Path(generated_root)
    user_gen_dirs = {}
    
    print(f"🔍 在 {generated_root} 中查找生成图像...")
    
    for user_id in user_ids:
        # 尝试多种可能的目录格式
        possible_dirs = [
            generated_root / f"user_{user_id:02d}",  # user_01
            generated_root / f"user_{user_id}",      # user_1
            generated_root / f"ID_{user_id}",        # ID_1
            generated_root / f"{user_id}",           # 1
        ]
        
        found = False
        for gen_dir in possible_dirs:
            if gen_dir.exists() and gen_dir.is_dir():
                # 检查是否有图像文件
                image_files = list(gen_dir.glob("*.png")) + list(gen_dir.glob("*.jpg"))
                if image_files:
                    user_gen_dirs[user_id] = str(gen_dir)
                    print(f"  找到用户 {user_id}: {gen_dir} ({len(image_files)} 张图像)")
                    found = True
                    break
        
        if not found:
            print(f"  ❌ 未找到用户 {user_id} 的生成图像")
    
    print(f"✅ 找到 {len(user_gen_dirs)} 个用户的生成图像")
    return user_gen_dirs

def validate_all_users(
    user_ids: List[int],
    classifiers_dir: str,
    generated_root: str,
    output_dir: str,
    confidence_threshold: float = 0.8
):
    """
    验证所有用户的生成图像
    
    Args:
        user_ids: 用户ID列表
        classifiers_dir: 分类器目录
        generated_root: 生成图像根目录
        output_dir: 输出目录
        confidence_threshold: 置信度阈值
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找生成图像
    user_gen_dirs = find_generated_images(generated_root, user_ids)
    
    if not user_gen_dirs:
        print("❌ 未找到任何生成图像，退出验证")
        return
    
    # 初始化验证系统
    validation_system = UserValidationSystem()
    
    # 加载分类器
    classifiers_path = Path(classifiers_dir)
    loaded_classifiers = []
    
    for user_id in user_ids:
        classifier_file = classifiers_path / f"user_{user_id:02d}_classifier.pth"
        
        if classifier_file.exists():
            try:
                validation_system.load_classifier(user_id, str(classifier_file))
                loaded_classifiers.append(user_id)
                print(f"✅ 加载用户 {user_id} 分类器")
            except Exception as e:
                print(f"❌ 加载用户 {user_id} 分类器失败: {e}")
        else:
            print(f"❌ 用户 {user_id} 分类器文件不存在: {classifier_file}")
    
    if not loaded_classifiers:
        print("❌ 未成功加载任何分类器，退出验证")
        return
    
    # 验证每个用户的生成图像
    all_results = []
    
    for user_id in loaded_classifiers:
        if user_id not in user_gen_dirs:
            print(f"⚠️  用户 {user_id} 没有生成图像，跳过验证")
            continue
        
        print(f"\n{'='*50}")
        print(f"🔍 验证用户 {user_id} 的生成图像")
        print(f"{'='*50}")
        
        try:
            # 验证生成图像
            result = validation_system.validate_generated_images(
                user_id=user_id,
                generated_images_dir=user_gen_dirs[user_id],
                confidence_threshold=confidence_threshold
            )
            
            all_results.append(result)
            
            # 保存单个用户结果
            user_result_path = output_path / f"user_{user_id:02d}_validation.json"
            with open(user_result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
        except Exception as e:
            print(f"❌ 用户 {user_id} 验证失败: {e}")
            continue
    
    if not all_results:
        print("❌ 没有成功验证任何用户")
        return
    
    # 生成总体报告
    print(f"\n{'='*50}")
    print("📊 生成验证报告")
    print(f"{'='*50}")
    
    # 保存所有结果
    all_results_path = output_path / "all_validation_results.json"
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 生成文本报告
    report_path = output_path / "validation_report.md"
    report_text = validation_system.generate_validation_report(all_results, str(report_path))
    
    # 打印总结
    print("\n📋 验证总结:")
    total_images = sum(r['total_images'] for r in all_results)
    total_success = sum(r['success_count'] for r in all_results)
    overall_success_rate = total_success / total_images if total_images > 0 else 0
    
    print(f"  总图像数: {total_images}")
    print(f"  成功图像数: {total_success}")
    print(f"  总体成功率: {overall_success_rate:.1%}")
    print(f"  置信度阈值: {confidence_threshold}")
    
    print(f"\n各用户详细结果:")
    for result in all_results:
        user_id = result['user_id']
        success_rate = result['success_rate']
        avg_confidence = result['avg_confidence']
        print(f"  用户 {user_id:2d}: {result['success_count']:2d}/{result['total_images']:2d} ({success_rate:.1%}) 平均置信度: {avg_confidence:.3f}")
    
    # 评估整体效果
    print(f"\n🎯 效果评估:")
    if overall_success_rate >= 0.8:
        print("🎉 优秀！生成图像很好地保持了用户特征")
    elif overall_success_rate >= 0.6:
        print("✅ 良好！生成图像较好地保持了用户特征")
    elif overall_success_rate >= 0.4:
        print("⚠️  一般！生成图像部分保持了用户特征，可能需要改进")
    else:
        print("❌ 较差！生成图像未能很好保持用户特征，需要重新训练或调整")
    
    print(f"\n📁 详细结果保存在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="验证生成图像的用户特征")
    
    # 输入参数
    parser.add_argument("--user_ids", type=int, nargs="+", required=True,
                       help="要验证的用户ID列表")
    parser.add_argument("--classifiers_dir", type=str, required=True,
                       help="分类器目录 (包含训练好的.pth文件)")
    parser.add_argument("--generated_root", type=str, required=True,
                       help="生成图像根目录 (包含user_01, user_02等子目录)")
    parser.add_argument("--output_dir", type=str, default="./validation_results",
                       help="输出目录")
    
    # 验证参数
    parser.add_argument("--confidence_threshold", type=float, default=0.8,
                       help="置信度阈值 (>0.8算成功)")
    
    args = parser.parse_args()
    
    print("🔍 生成图像用户特征验证")
    print("=" * 50)
    print(f"用户ID列表: {args.user_ids}")
    print(f"分类器目录: {args.classifiers_dir}")
    print(f"生成图像根目录: {args.generated_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"置信度阈值: {args.confidence_threshold}")
    print("=" * 50)
    
    validate_all_users(
        user_ids=args.user_ids,
        classifiers_dir=args.classifiers_dir,
        generated_root=args.generated_root,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold
    )

if __name__ == "__main__":
    main()
