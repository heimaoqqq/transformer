#!/usr/bin/env python3
"""
诊断分类器问题
检查分类器是否正确训练和预测
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from validation.user_classifier import UserValidationSystem

def test_classifier_on_training_data(
    user_id: int,
    classifier_path: str,
    real_data_root: str,
    num_test_samples: int = 20
):
    """
    测试分类器在训练数据上的表现
    """
    print(f"🔍 测试用户 {user_id} 分类器在训练数据上的表现")
    
    # 初始化验证系统
    validation_system = UserValidationSystem()
    
    # 加载分类器
    try:
        validation_system.load_classifier(user_id, classifier_path)
        print(f"✅ 成功加载分类器")
    except Exception as e:
        print(f"❌ 加载分类器失败: {e}")
        return
    
    # 查找用户数据
    data_root = Path(real_data_root)
    target_user_dir = None
    other_user_dirs = []
    
    for user_dir in data_root.iterdir():
        if user_dir.is_dir():
            dir_name = user_dir.name
            if dir_name == f"ID_{user_id}":
                target_user_dir = str(user_dir)
            elif dir_name.startswith('ID_'):
                try:
                    other_user_id = int(dir_name.split('_')[1])
                    if other_user_id != user_id:
                        other_user_dirs.append(str(user_dir))
                except ValueError:
                    continue
    
    if target_user_dir is None:
        print(f"❌ 未找到用户 {user_id} 的数据目录")
        return
    
    print(f"✅ 找到用户数据目录: {target_user_dir}")
    print(f"✅ 找到 {len(other_user_dirs)} 个其他用户目录")
    
    # 测试正样本 (该用户的真实图像)
    print(f"\n📊 测试正样本 (用户 {user_id} 的真实图像):")
    
    user_images = list(Path(target_user_dir).glob("*.png")) + list(Path(target_user_dir).glob("*.jpg"))
    test_positive = user_images[:num_test_samples]
    
    positive_confidences = []
    for img_path in test_positive:
        try:
            # 加载和预处理图像
            image = Image.open(img_path).convert('RGB')
            image_tensor = validation_system.transform(image).unsqueeze(0).to(validation_system.device)
            
            # 预测
            with torch.no_grad():
                model = validation_system.classifiers[user_id]
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence = probabilities[0, 1].item()  # 正类置信度
                positive_confidences.append(confidence)
                
        except Exception as e:
            print(f"  ⚠️  处理图像失败 {img_path}: {e}")
    
    if positive_confidences:
        avg_pos_conf = np.mean(positive_confidences)
        print(f"  正样本数量: {len(positive_confidences)}")
        print(f"  平均置信度: {avg_pos_conf:.3f}")
        print(f"  置信度范围: [{min(positive_confidences):.3f}, {max(positive_confidences):.3f}]")
        print(f"  高置信度(>0.8)比例: {sum(1 for c in positive_confidences if c > 0.8)/len(positive_confidences):.1%}")
    
    # 测试负样本 (其他用户的图像)
    print(f"\n📊 测试负样本 (其他用户的图像):")
    
    negative_confidences = []
    for other_dir in other_user_dirs[:3]:  # 只测试前3个其他用户
        other_images = list(Path(other_dir).glob("*.png")) + list(Path(other_dir).glob("*.jpg"))
        test_negative = other_images[:5]  # 每个用户测试5张
        
        for img_path in test_negative:
            try:
                # 加载和预处理图像
                image = Image.open(img_path).convert('RGB')
                image_tensor = validation_system.transform(image).unsqueeze(0).to(validation_system.device)
                
                # 预测
                with torch.no_grad():
                    model = validation_system.classifiers[user_id]
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence = probabilities[0, 1].item()  # 正类置信度
                    negative_confidences.append(confidence)
                    
            except Exception as e:
                print(f"  ⚠️  处理图像失败 {img_path}: {e}")
    
    if negative_confidences:
        avg_neg_conf = np.mean(negative_confidences)
        print(f"  负样本数量: {len(negative_confidences)}")
        print(f"  平均置信度: {avg_neg_conf:.3f}")
        print(f"  置信度范围: [{min(negative_confidences):.3f}, {max(negative_confidences):.3f}]")
        print(f"  低置信度(<0.2)比例: {sum(1 for c in negative_confidences if c < 0.2)/len(negative_confidences):.1%}")
    
    # 分析分类器性能
    print(f"\n📈 分类器性能分析:")
    if positive_confidences and negative_confidences:
        separation = avg_pos_conf - avg_neg_conf
        print(f"  正负样本置信度差异: {separation:.3f}")
        
        if separation > 0.5:
            print(f"  ✅ 分类器区分度良好")
        elif separation > 0.2:
            print(f"  ⚠️  分类器区分度一般")
        else:
            print(f"  ❌ 分类器区分度很差")
            
        # 检查是否有明显的分类边界
        pos_high = sum(1 for c in positive_confidences if c > 0.8)
        neg_low = sum(1 for c in negative_confidences if c < 0.2)
        
        print(f"  正样本高置信度(>0.8): {pos_high}/{len(positive_confidences)} ({pos_high/len(positive_confidences):.1%})")
        print(f"  负样本低置信度(<0.2): {neg_low}/{len(negative_confidences)} ({neg_low/len(negative_confidences):.1%})")
        
        if pos_high/len(positive_confidences) > 0.7 and neg_low/len(negative_confidences) > 0.7:
            print(f"  ✅ 分类器训练良好")
        else:
            print(f"  ❌ 分类器可能训练不足或过拟合")

def test_classifier_on_generated_images(
    user_id: int,
    classifier_path: str,
    generated_images_dir: str
):
    """
    测试分类器在生成图像上的表现，并显示一些图像
    """
    print(f"\n🎨 测试用户 {user_id} 分类器在生成图像上的表现")
    
    # 初始化验证系统
    validation_system = UserValidationSystem()
    
    # 加载分类器
    validation_system.load_classifier(user_id, classifier_path)
    
    # 查找生成图像
    gen_dir = Path(generated_images_dir)
    if not gen_dir.exists():
        print(f"❌ 生成图像目录不存在: {gen_dir}")
        return
    
    image_files = list(gen_dir.glob("*.png")) + list(gen_dir.glob("*.jpg"))
    if not image_files:
        print(f"❌ 未找到生成图像")
        return
    
    print(f"✅ 找到 {len(image_files)} 张生成图像")
    
    # 测试所有生成图像
    generated_confidences = []
    image_details = []
    
    for img_path in image_files:
        try:
            # 加载和预处理图像
            image = Image.open(img_path).convert('RGB')
            image_tensor = validation_system.transform(image).unsqueeze(0).to(validation_system.device)
            
            # 预测
            with torch.no_grad():
                model = validation_system.classifiers[user_id]
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence = probabilities[0, 1].item()  # 正类置信度
                generated_confidences.append(confidence)
                image_details.append((img_path.name, confidence))
                
        except Exception as e:
            print(f"  ⚠️  处理图像失败 {img_path}: {e}")
    
    if generated_confidences:
        avg_gen_conf = np.mean(generated_confidences)
        print(f"  生成图像数量: {len(generated_confidences)}")
        print(f"  平均置信度: {avg_gen_conf:.3f}")
        print(f"  置信度范围: [{min(generated_confidences):.3f}, {max(generated_confidences):.3f}]")
        
        # 显示每张图像的置信度
        print(f"\n📋 各图像置信度详情:")
        for name, conf in sorted(image_details, key=lambda x: x[1], reverse=True):
            status = "✅" if conf > 0.8 else "⚠️" if conf > 0.3 else "❌"
            print(f"    {status} {name}: {conf:.3f}")
        
        # 分析置信度分布
        high_conf = sum(1 for c in generated_confidences if c > 0.8)
        medium_conf = sum(1 for c in generated_confidences if 0.3 < c <= 0.8)
        low_conf = sum(1 for c in generated_confidences if c <= 0.3)
        
        print(f"\n📊 置信度分布:")
        print(f"  高置信度 (>0.8): {high_conf}/{len(generated_confidences)} ({high_conf/len(generated_confidences):.1%})")
        print(f"  中置信度 (0.3-0.8): {medium_conf}/{len(generated_confidences)} ({medium_conf/len(generated_confidences):.1%})")
        print(f"  低置信度 (≤0.3): {low_conf}/{len(generated_confidences)} ({low_conf/len(generated_confidences):.1%})")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="诊断分类器问题")
    parser.add_argument("--user_id", type=int, required=True, help="用户ID")
    parser.add_argument("--classifier_path", type=str, required=True, help="分类器路径")
    parser.add_argument("--real_data_root", type=str, required=True, help="真实数据根目录")
    parser.add_argument("--generated_images_dir", type=str, help="生成图像目录")
    parser.add_argument("--num_test_samples", type=int, default=20, help="测试样本数量")
    
    args = parser.parse_args()
    
    print("🔍 分类器诊断工具")
    print("=" * 50)
    
    # 测试分类器在训练数据上的表现
    test_classifier_on_training_data(
        user_id=args.user_id,
        classifier_path=args.classifier_path,
        real_data_root=args.real_data_root,
        num_test_samples=args.num_test_samples
    )
    
    # 如果提供了生成图像目录，也测试生成图像
    if args.generated_images_dir:
        test_classifier_on_generated_images(
            user_id=args.user_id,
            classifier_path=args.classifier_path,
            generated_images_dir=args.generated_images_dir
        )
    
    print("\n" + "=" * 50)
    print("🎯 诊断建议:")
    print("1. 如果分类器在真实数据上表现差，需要重新训练")
    print("2. 如果分类器在真实数据上表现好，但生成图像上表现差，说明生成图像质量有问题")
    print("3. 检查图像预处理是否一致")
    print("4. 检查用户ID映射是否正确")

if __name__ == "__main__":
    main()
