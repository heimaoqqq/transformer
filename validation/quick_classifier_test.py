#!/usr/bin/env python3
"""
快速测试修复后的分类器
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from validation.user_classifier import UserValidationSystem

def quick_test_classifier():
    """快速测试分类器的基本功能"""
    print("🔧 快速测试修复后的分类器")
    
    # 创建验证系统
    validation_system = UserValidationSystem()
    
    # 测试分类器创建
    print("\n1️⃣ 测试分类器创建:")
    try:
        from validation.user_classifier import UserClassifier
        model = UserClassifier(num_classes=2, pretrained=True)
        print("✅ 分类器创建成功")
        
        # 测试forward
        test_input = torch.randn(1, 3, 128, 128)  # 修复后的尺寸
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ Forward测试成功，输出形状: {output.shape}")
            
            # 测试softmax
            probabilities = torch.softmax(output, dim=1)
            print(f"✅ Softmax测试成功，概率: {probabilities}")
            
    except Exception as e:
        print(f"❌ 分类器测试失败: {e}")
        return False
    
    # 测试图像预处理
    print("\n2️⃣ 测试图像预处理:")
    try:
        # 创建测试图像
        test_image = Image.new('RGB', (256, 256), (128, 128, 128))
        
        # 应用变换
        transformed = validation_system.transform(test_image)
        print(f"✅ 图像变换成功，输出形状: {transformed.shape}")
        print(f"✅ 像素值范围: [{transformed.min():.3f}, {transformed.max():.3f}]")
        
        # 测试批处理
        batch = transformed.unsqueeze(0)
        with torch.no_grad():
            output = model(batch)
            probabilities = torch.softmax(output, dim=1)
            confidence = probabilities[0, 1].item()
            print(f"✅ 批处理测试成功，置信度: {confidence:.3f}")
            
    except Exception as e:
        print(f"❌ 图像预处理测试失败: {e}")
        return False
    
    print("\n✅ 所有基本测试通过！")
    return True

def test_on_real_data(data_dir: str, user_id: int = 1):
    """在真实数据上测试"""
    print(f"\n3️⃣ 在真实数据上测试 (用户 {user_id}):")
    
    data_path = Path(data_dir)
    user_dir = data_path / f"ID_{user_id}"
    
    if not user_dir.exists():
        print(f"❌ 用户目录不存在: {user_dir}")
        return False
    
    # 找到一些图像
    image_files = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))
    if not image_files:
        print(f"❌ 未找到图像文件")
        return False
    
    print(f"✅ 找到 {len(image_files)} 张图像")
    
    # 创建验证系统和模型
    validation_system = UserValidationSystem()
    from validation.user_classifier import UserClassifier
    model = UserClassifier(num_classes=2, pretrained=True)
    model.eval()
    
    # 测试前几张图像
    test_images = image_files[:5]
    confidences = []
    
    for img_path in test_images:
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            print(f"  原始图像尺寸: {image.size}")
            
            # 预处理
            image_tensor = validation_system.transform(image).unsqueeze(0)
            print(f"  预处理后形状: {image_tensor.shape}")
            
            # 预测
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence = probabilities[0, 1].item()
                confidences.append(confidence)
                
                print(f"  {img_path.name}: 置信度 {confidence:.3f}")
                
        except Exception as e:
            print(f"  ❌ 处理 {img_path.name} 失败: {e}")
    
    if confidences:
        avg_conf = np.mean(confidences)
        print(f"\n📊 真实数据测试结果:")
        print(f"  测试图像数: {len(confidences)}")
        print(f"  平均置信度: {avg_conf:.3f}")
        print(f"  置信度范围: [{min(confidences):.3f}, {max(confidences):.3f}]")
        
        # 注意：这是未训练的模型，所以置信度应该接近随机(0.5)
        if 0.3 < avg_conf < 0.7:
            print(f"  ✅ 未训练模型的随机输出正常")
        else:
            print(f"  ⚠️  未训练模型的输出异常")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="快速测试分类器修复")
    parser.add_argument("--data_dir", type=str, help="数据目录 (可选)")
    parser.add_argument("--user_id", type=int, default=1, help="测试用户ID")
    
    args = parser.parse_args()
    
    print("🔧 分类器修复验证")
    print("=" * 40)
    
    # 基本功能测试
    basic_ok = quick_test_classifier()
    
    # 真实数据测试 (如果提供)
    if args.data_dir and basic_ok:
        real_data_ok = test_on_real_data(args.data_dir, args.user_id)
    
    print("\n" + "=" * 40)
    if basic_ok:
        print("✅ 分类器修复验证通过")
        print("💡 主要修复:")
        print("   1. 简化了forward函数")
        print("   2. 修复了图像尺寸 (64→128)")
        print("   3. 移除了不一致的归一化")
        print("\n🚀 现在可以重新训练分类器了")
    else:
        print("❌ 分类器修复验证失败")

if __name__ == "__main__":
    main()
