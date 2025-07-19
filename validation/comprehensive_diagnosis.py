#!/usr/bin/env python3
"""
全面诊断验证失败的根本原因
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from validation.user_classifier import UserValidationSystem

def analyze_real_data_diversity(data_root: str, target_user_id: int, num_samples: int = 10):
    """
    分析真实数据的多样性和用户间差异
    """
    print(f"📊 分析真实数据的用户间差异")
    
    data_path = Path(data_root)
    
    # 收集所有用户的样本
    user_samples = {}
    for user_dir in data_path.iterdir():
        if user_dir.is_dir() and user_dir.name.startswith('ID_'):
            try:
                user_id = int(user_dir.name.split('_')[1])
                images = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))
                if images:
                    user_samples[user_id] = images[:num_samples]
            except ValueError:
                continue
    
    print(f"  找到 {len(user_samples)} 个用户")
    
    # 分析目标用户与其他用户的差异
    if target_user_id not in user_samples:
        print(f"❌ 目标用户 {target_user_id} 不存在")
        return False
    
    target_images = user_samples[target_user_id]
    print(f"  目标用户 {target_user_id}: {len(target_images)} 张图像")
    
    # 简单的像素级差异分析
    validation_system = UserValidationSystem()
    
    def load_and_process(img_path):
        try:
            image = Image.open(img_path).convert('RGB')
            tensor = validation_system.transform(image)
            return tensor.numpy().flatten()
        except:
            return None
    
    # 加载目标用户图像
    target_vectors = []
    for img_path in target_images:
        vec = load_and_process(img_path)
        if vec is not None:
            target_vectors.append(vec)
    
    if not target_vectors:
        print(f"❌ 无法加载目标用户图像")
        return False
    
    target_vectors = np.array(target_vectors)
    target_mean = np.mean(target_vectors, axis=0)
    
    # 计算与其他用户的差异
    user_distances = {}
    for other_user_id, other_images in user_samples.items():
        if other_user_id == target_user_id:
            continue
            
        other_vectors = []
        for img_path in other_images[:5]:  # 每个用户取5张
            vec = load_and_process(img_path)
            if vec is not None:
                other_vectors.append(vec)
        
        if other_vectors:
            other_vectors = np.array(other_vectors)
            other_mean = np.mean(other_vectors, axis=0)
            distance = np.linalg.norm(target_mean - other_mean)
            user_distances[other_user_id] = distance
    
    if user_distances:
        avg_distance = np.mean(list(user_distances.values()))
        min_distance = min(user_distances.values())
        max_distance = max(user_distances.values())
        
        print(f"  用户间像素级差异:")
        print(f"    平均距离: {avg_distance:.2f}")
        print(f"    最小距离: {min_distance:.2f}")
        print(f"    最大距离: {max_distance:.2f}")
        
        if avg_distance < 50:
            print(f"  ⚠️  用户间差异很小，可能难以区分")
        elif avg_distance < 100:
            print(f"  📊 用户间差异中等")
        else:
            print(f"  ✅ 用户间差异明显")
    
    return True

def test_classifier_on_real_data(
    user_id: int,
    classifier_path: str,
    data_root: str,
    num_test_per_user: int = 20
):
    """
    详细测试分类器在真实数据上的表现
    """
    print(f"\n🎯 详细测试分类器在真实数据上的表现")
    
    # 加载分类器
    validation_system = UserValidationSystem()
    try:
        validation_system.load_classifier(user_id, classifier_path)
        model = validation_system.classifiers[user_id]
        print(f"✅ 成功加载分类器")
    except Exception as e:
        print(f"❌ 加载分类器失败: {e}")
        return False
    
    # 收集测试数据
    data_path = Path(data_root)
    positive_samples = []
    negative_samples = []
    
    # 正样本：目标用户
    target_dir = data_path / f"ID_{user_id}"
    if target_dir.exists():
        images = list(target_dir.glob("*.png")) + list(target_dir.glob("*.jpg"))
        positive_samples = images[:num_test_per_user]
    
    # 负样本：其他用户
    for user_dir in data_path.iterdir():
        if user_dir.is_dir() and user_dir.name.startswith('ID_'):
            try:
                other_user_id = int(user_dir.name.split('_')[1])
                if other_user_id != user_id:
                    images = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))
                    negative_samples.extend(images[:5])  # 每个其他用户取5张
            except ValueError:
                continue
    
    print(f"  正样本: {len(positive_samples)} 张")
    print(f"  负样本: {len(negative_samples)} 张")
    
    if len(positive_samples) < 5 or len(negative_samples) < 5:
        print(f"❌ 测试样本不足")
        return False
    
    # 测试分类器
    def predict_batch(image_paths, label):
        predictions = []
        confidences = []
        labels = []
        
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = validation_system.transform(image).unsqueeze(0).to(validation_system.device)
                
                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence = probabilities[0, 1].item()
                    prediction = 1 if confidence > 0.5 else 0
                    
                    predictions.append(prediction)
                    confidences.append(confidence)
                    labels.append(label)
                    
            except Exception as e:
                print(f"    处理失败 {img_path}: {e}")
        
        return predictions, confidences, labels
    
    # 测试正样本
    print(f"  测试正样本...")
    pos_preds, pos_confs, pos_labels = predict_batch(positive_samples, 1)
    
    # 测试负样本
    print(f"  测试负样本...")
    neg_preds, neg_confs, neg_labels = predict_batch(negative_samples, 0)
    
    # 合并结果
    all_preds = pos_preds + neg_preds
    all_confs = pos_confs + neg_confs
    all_labels = pos_labels + neg_labels
    
    if len(all_preds) < 10:
        print(f"❌ 有效预测太少")
        return False
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    
    pos_acc = accuracy_score(pos_labels, pos_preds) if pos_preds else 0
    neg_acc = accuracy_score(neg_labels, neg_preds) if neg_preds else 0
    
    avg_pos_conf = np.mean(pos_confs) if pos_confs else 0
    avg_neg_conf = np.mean(neg_confs) if neg_confs else 0
    
    print(f"\n📊 分类器在真实数据上的表现:")
    print(f"  总体准确率: {accuracy:.3f}")
    print(f"  正样本准确率: {pos_acc:.3f}")
    print(f"  负样本准确率: {neg_acc:.3f}")
    print(f"  正样本平均置信度: {avg_pos_conf:.3f}")
    print(f"  负样本平均置信度: {avg_neg_conf:.3f}")
    print(f"  置信度差异: {avg_pos_conf - avg_neg_conf:.3f}")
    
    # 判断分类器质量
    if accuracy > 0.8 and abs(avg_pos_conf - avg_neg_conf) > 0.3:
        print(f"  ✅ 分类器在真实数据上表现良好")
        return True
    elif accuracy > 0.6:
        print(f"  ⚠️  分类器在真实数据上表现一般")
        return True
    else:
        print(f"  ❌ 分类器在真实数据上表现很差")
        return False

def analyze_generated_vs_real(
    user_id: int,
    real_data_root: str,
    generated_images_dir: str
):
    """
    分析生成图像与真实图像的差异
    """
    print(f"\n🔍 分析生成图像与真实图像的差异")
    
    validation_system = UserValidationSystem()
    
    # 加载真实图像
    real_dir = Path(real_data_root) / f"ID_{user_id}"
    if not real_dir.exists():
        print(f"❌ 真实图像目录不存在")
        return False
    
    real_images = list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg"))
    
    # 加载生成图像
    gen_dir = Path(generated_images_dir)
    if not gen_dir.exists():
        print(f"❌ 生成图像目录不存在")
        return False
    
    gen_images = list(gen_dir.glob("*.png")) + list(gen_dir.glob("*.jpg"))
    
    print(f"  真实图像: {len(real_images)} 张")
    print(f"  生成图像: {len(gen_images)} 张")
    
    def load_and_process(img_path):
        try:
            image = Image.open(img_path).convert('RGB')
            tensor = validation_system.transform(image)
            return tensor.numpy().flatten()
        except:
            return None
    
    # 处理真实图像
    real_vectors = []
    for img_path in real_images[:20]:  # 取前20张
        vec = load_and_process(img_path)
        if vec is not None:
            real_vectors.append(vec)
    
    # 处理生成图像
    gen_vectors = []
    for img_path in gen_images:
        vec = load_and_process(img_path)
        if vec is not None:
            gen_vectors.append(vec)
    
    if not real_vectors or not gen_vectors:
        print(f"❌ 无法加载图像")
        return False
    
    real_vectors = np.array(real_vectors)
    gen_vectors = np.array(gen_vectors)
    
    # 计算统计差异
    real_mean = np.mean(real_vectors, axis=0)
    gen_mean = np.mean(gen_vectors, axis=0)
    
    mean_distance = np.linalg.norm(real_mean - gen_mean)
    
    # 计算方差
    real_var = np.var(real_vectors)
    gen_var = np.var(gen_vectors)
    
    print(f"  像素级分析:")
    print(f"    真实图像均值距离生成图像均值: {mean_distance:.2f}")
    print(f"    真实图像方差: {real_var:.2f}")
    print(f"    生成图像方差: {gen_var:.2f}")
    
    if mean_distance > 100:
        print(f"  ❌ 生成图像与真实图像差异很大")
    elif mean_distance > 50:
        print(f"  ⚠️  生成图像与真实图像有一定差异")
    else:
        print(f"  ✅ 生成图像与真实图像相似")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="全面诊断验证失败原因")
    parser.add_argument("--user_id", type=int, required=True, help="用户ID")
    parser.add_argument("--real_data_root", type=str, required=True, help="真实数据根目录")
    parser.add_argument("--classifier_path", type=str, required=True, help="分类器路径")
    parser.add_argument("--generated_images_dir", type=str, help="生成图像目录")
    
    args = parser.parse_args()
    
    print("🔍 全面诊断验证失败原因")
    print("=" * 60)
    
    # 1. 分析真实数据多样性
    diversity_ok = analyze_real_data_diversity(args.real_data_root, args.user_id)
    
    # 2. 测试分类器在真实数据上的表现
    classifier_ok = test_classifier_on_real_data(
        args.user_id, args.classifier_path, args.real_data_root
    )
    
    # 3. 分析生成图像与真实图像的差异
    if args.generated_images_dir:
        similarity_ok = analyze_generated_vs_real(
            args.user_id, args.real_data_root, args.generated_images_dir
        )
    
    print("\n" + "=" * 60)
    print("🎯 诊断结论:")
    
    if not diversity_ok:
        print("❌ 数据问题：用户间差异太小或数据质量有问题")
    elif not classifier_ok:
        print("❌ 分类器问题：分类器在真实数据上表现差")
    elif args.generated_images_dir and not similarity_ok:
        print("❌ 生成问题：生成图像与真实图像差异太大")
    else:
        print("⚠️  需要进一步分析，可能是条件扩散模型没有学到用户特征")

if __name__ == "__main__":
    main()
