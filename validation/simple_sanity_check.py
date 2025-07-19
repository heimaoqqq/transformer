#!/usr/bin/env python3
"""
简单的合理性检查
检查最基本的问题
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

def check_basic_sanity():
    """基本合理性检查"""
    print("🔍 基本合理性检查")
    
    # 1. 检查分类器是否能区分明显不同的图像
    print("\n1️⃣ 测试分类器是否能区分明显不同的图像:")
    
    validation_system = UserValidationSystem()
    from validation.user_classifier import UserClassifier
    
    # 创建一个新的分类器
    model = UserClassifier(num_classes=2, pretrained=True)
    model.eval()
    
    # 创建两个明显不同的测试图像
    # 图像1：全黑
    black_image = Image.new('RGB', (128, 128), (0, 0, 0))
    black_tensor = validation_system.transform(black_image).unsqueeze(0)
    
    # 图像2：全白
    white_image = Image.new('RGB', (128, 128), (255, 255, 255))
    white_tensor = validation_system.transform(white_image).unsqueeze(0)
    
    # 图像3：随机噪声
    noise_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    noise_image = Image.fromarray(noise_array)
    noise_tensor = validation_system.transform(noise_image).unsqueeze(0)
    
    with torch.no_grad():
        # 测试黑色图像
        output_black = model(black_tensor)
        prob_black = torch.softmax(output_black, dim=1)
        conf_black = prob_black[0, 1].item()
        
        # 测试白色图像
        output_white = model(white_tensor)
        prob_white = torch.softmax(output_white, dim=1)
        conf_white = prob_white[0, 1].item()
        
        # 测试噪声图像
        output_noise = model(noise_tensor)
        prob_noise = torch.softmax(output_noise, dim=1)
        conf_noise = prob_noise[0, 1].item()
    
    print(f"  黑色图像置信度: {conf_black:.3f}")
    print(f"  白色图像置信度: {conf_white:.3f}")
    print(f"  噪声图像置信度: {conf_noise:.3f}")
    
    # 检查是否有明显差异
    max_conf = max(conf_black, conf_white, conf_noise)
    min_conf = min(conf_black, conf_white, conf_noise)
    diff = max_conf - min_conf
    
    print(f"  置信度差异: {diff:.3f}")
    
    if diff > 0.1:
        print(f"  ✅ 分类器能区分不同图像")
    else:
        print(f"  ❌ 分类器输出几乎相同，可能有问题")
    
    return diff > 0.1

def check_real_data_loading(data_root: str, user_id: int = 1):
    """检查真实数据加载"""
    print(f"\n2️⃣ 检查真实数据加载 (用户 {user_id}):")
    
    data_path = Path(data_root)
    user_dir = data_path / f"ID_{user_id}"
    
    if not user_dir.exists():
        print(f"❌ 用户目录不存在: {user_dir}")
        return False
    
    # 查找图像文件
    image_files = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))
    print(f"  找到 {len(image_files)} 张图像")
    
    if len(image_files) == 0:
        print(f"❌ 未找到图像文件")
        return False
    
    # 测试加载前几张图像
    validation_system = UserValidationSystem()
    loaded_count = 0
    
    for i, img_path in enumerate(image_files[:5]):
        try:
            # 加载原始图像
            image = Image.open(img_path).convert('RGB')
            print(f"  图像 {i+1}: {image.size} -> ", end="")
            
            # 预处理
            tensor = validation_system.transform(image)
            print(f"{tensor.shape}, 范围: [{tensor.min():.3f}, {tensor.max():.3f}]")
            
            loaded_count += 1
            
        except Exception as e:
            print(f"  图像 {i+1}: 加载失败 - {e}")
    
    print(f"  成功加载: {loaded_count}/5")
    return loaded_count >= 3

def check_generated_images(gen_dir: str):
    """检查生成图像"""
    print(f"\n3️⃣ 检查生成图像:")
    
    gen_path = Path(gen_dir)
    if not gen_path.exists():
        print(f"❌ 生成图像目录不存在: {gen_path}")
        return False
    
    image_files = list(gen_path.glob("*.png")) + list(gen_path.glob("*.jpg"))
    print(f"  找到 {len(image_files)} 张生成图像")
    
    if len(image_files) == 0:
        print(f"❌ 未找到生成图像")
        return False
    
    # 检查生成图像的基本属性
    validation_system = UserValidationSystem()
    
    for i, img_path in enumerate(image_files[:3]):
        try:
            image = Image.open(img_path).convert('RGB')
            tensor = validation_system.transform(image)
            
            # 检查是否是纯色图像（可能生成失败）
            std = tensor.std().item()
            mean = tensor.mean().item()
            
            print(f"  图像 {i+1}: {image.size} -> 标准差: {std:.3f}, 均值: {mean:.3f}")
            
            if std < 0.01:
                print(f"    ⚠️  图像可能是纯色（生成失败）")
            elif std > 0.3:
                print(f"    ✅ 图像有丰富内容")
            else:
                print(f"    📊 图像内容正常")
                
        except Exception as e:
            print(f"  图像 {i+1}: 处理失败 - {e}")
    
    return True

def check_classifier_training_history(history_file: str):
    """检查分类器训练历史"""
    print(f"\n4️⃣ 检查分类器训练历史:")
    
    history_path = Path(history_file)
    if not history_path.exists():
        print(f"❌ 训练历史文件不存在: {history_path}")
        return False
    
    try:
        import json
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        if 'val_acc' in history and history['val_acc']:
            final_val_acc = history['val_acc'][-1]
            best_val_acc = max(history['val_acc'])
            
            print(f"  最终验证准确率: {final_val_acc:.3f}")
            print(f"  最佳验证准确率: {best_val_acc:.3f}")
            
            if best_val_acc > 0.8:
                print(f"  ✅ 分类器训练良好")
                return True
            elif best_val_acc > 0.6:
                print(f"  ⚠️  分类器训练一般")
                return True
            else:
                print(f"  ❌ 分类器训练很差")
                return False
        else:
            print(f"❌ 训练历史格式异常")
            return False
            
    except Exception as e:
        print(f"❌ 读取训练历史失败: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="简单合理性检查")
    parser.add_argument("--data_root", type=str, help="真实数据根目录")
    parser.add_argument("--user_id", type=int, default=1, help="用户ID")
    parser.add_argument("--generated_dir", type=str, help="生成图像目录")
    parser.add_argument("--history_file", type=str, help="训练历史文件")
    
    args = parser.parse_args()
    
    print("🔍 简单合理性检查")
    print("=" * 40)
    
    # 基本检查
    basic_ok = check_basic_sanity()
    
    # 数据检查
    data_ok = True
    if args.data_root:
        data_ok = check_real_data_loading(args.data_root, args.user_id)
    
    # 生成图像检查
    gen_ok = True
    if args.generated_dir:
        gen_ok = check_generated_images(args.generated_dir)
    
    # 训练历史检查
    history_ok = True
    if args.history_file:
        history_ok = check_classifier_training_history(args.history_file)
    
    print("\n" + "=" * 40)
    print("📊 检查总结:")
    print(f"  基本功能: {'✅' if basic_ok else '❌'}")
    print(f"  数据加载: {'✅' if data_ok else '❌'}")
    print(f"  生成图像: {'✅' if gen_ok else '❌'}")
    print(f"  训练历史: {'✅' if history_ok else '❌'}")
    
    if not basic_ok:
        print("\n💡 建议: 分类器基本功能有问题，需要检查模型架构")
    elif not data_ok:
        print("\n💡 建议: 数据加载有问题，检查数据路径和格式")
    elif not gen_ok:
        print("\n💡 建议: 生成图像有问题，可能生成失败")
    elif not history_ok:
        print("\n💡 建议: 分类器训练不充分，需要重新训练")
    else:
        print("\n💡 建议: 基本检查都通过，问题可能更深层")

if __name__ == "__main__":
    main()
