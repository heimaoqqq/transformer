#!/usr/bin/env python3
"""
简化的条件扩散验证脚本
专门验证31位用户的步态微多普勒时频图像条件扩散效果

使用方法:
1. 训练31个用户分类器 (每个用户一个二分类器: 是/不是该用户)
2. 生成指定用户的图像
3. 用对应的分类器验证生成图像是否包含该用户特征

核心思路:
- 如果条件扩散真的有效，生成的用户A图像应该被用户A的分类器识别为"是用户A"
- 同时被其他用户的分类器识别为"不是该用户"
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
from PIL import Image
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from validation.user_classifier import UserValidationSystem
import random

class SimpleConditionValidator:
    """简化的条件扩散验证器"""
    
    def __init__(self, data_dir: str, output_dir: str = "./validation_results", device: str = "auto"):
        """
        Args:
            data_dir: 数据目录，包含ID_1, ID_2, ..., ID_31等子目录
            output_dir: 输出目录
            device: 计算设备
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"🚀 设备: {self.device}")
        print(f"📊 数据集格式: 256×256彩色图像")
        print(f"📁 数据目录: {self.data_dir}")

        # 初始化验证系统 (针对256×256图像优化)
        self.validation_system = UserValidationSystem(device=str(self.device))

        # 扫描用户数据
        self.user_mapping = self._scan_users()
        self.num_users = len(self.user_mapping)

        print(f"📊 发现 {self.num_users} 个用户: {sorted(self.user_mapping.keys())}")

        # 验证数据集格式
        self._validate_dataset_format()
    
    def _scan_users(self) -> Dict[int, str]:
        """扫描用户目录"""
        user_mapping = {}
        
        for user_dir in self.data_dir.iterdir():
            if user_dir.is_dir() and user_dir.name.startswith('ID_'):
                try:
                    user_id = int(user_dir.name.split('_')[1])
                    image_files = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))
                    
                    if len(image_files) > 0:
                        user_mapping[user_id] = str(user_dir)
                        print(f"  用户 {user_id:2d}: {len(image_files):3d} 张图像")
                    else:
                        print(f"  ⚠️  用户 {user_id:2d}: 无图像文件")
                        
                except ValueError:
                    print(f"  ❌ 无效目录名: {user_dir.name}")
                    continue
        
        return user_mapping

    def _validate_dataset_format(self):
        """验证数据集格式"""
        print(f"\n🔍 验证数据集格式...")

        total_images = 0
        format_issues = []

        for user_id, user_dir in self.user_mapping.items():
            user_path = Path(user_dir)
            image_files = list(user_path.glob("*.png")) + list(user_path.glob("*.jpg"))

            if len(image_files) > 0:
                # 检查第一张图像的格式
                try:
                    sample_image = Image.open(image_files[0])
                    width, height = sample_image.size
                    mode = sample_image.mode

                    if width != 256 or height != 256:
                        format_issues.append(f"用户 {user_id}: 尺寸 {width}×{height} (期望 256×256)")

                    if mode != 'RGB':
                        format_issues.append(f"用户 {user_id}: 模式 {mode} (期望 RGB)")

                    total_images += len(image_files)

                except Exception as e:
                    format_issues.append(f"用户 {user_id}: 图像读取失败 - {e}")

        print(f"  总图像数: {total_images}")

        if format_issues:
            print(f"  ⚠️  格式问题:")
            for issue in format_issues[:5]:  # 只显示前5个问题
                print(f"    {issue}")
            if len(format_issues) > 5:
                print(f"    ... 还有 {len(format_issues) - 5} 个问题")
        else:
            print(f"  ✅ 数据集格式验证通过")

    def generate_test_images(self, model_dir: str, user_id: int, num_images: int = 50,
                           num_inference_steps: int = 50, guidance_scale: float = 1.0) -> str:
        """
        生成测试图像 (支持指导强度)

        Args:
            model_dir: 模型目录
            user_id: 用户ID
            num_images: 生成图像数量
            num_inference_steps: 推理步数
            guidance_scale: 指导强度 (1.0=纯条件, >1.0=CFG增强)

        Returns:
            生成图像目录路径
        """
        print(f"\n🎨 为用户 {user_id} 生成测试图像")
        print(f"  参数: {num_images}张, {num_inference_steps}步, 指导强度={guidance_scale}")

        generated_dir = self.output_dir / f"generated_user_{user_id:02d}"
        generated_dir.mkdir(exist_ok=True)

        # 构建生成命令
        cmd = [
            "python", "inference/generate.py",
            "--vae_path", f"{model_dir}/vae/final_model",
            "--unet_path", f"{model_dir}/unet/final_model",
            "--condition_encoder_path", f"{model_dir}/condition_encoder/final_model.pth",
            "--user_ids", str(user_id),
            "--num_images_per_user", str(num_images),
            "--num_inference_steps", str(num_inference_steps),
            "--output_dir", str(generated_dir),
            "--data_dir", str(self.data_dir),
            "--num_users", str(self.num_users)
        ]

        # 如果支持指导强度，添加参数
        if guidance_scale > 1.0:
            cmd.extend(["--guidance_scale", str(guidance_scale)])

        print(f"  命令: {' '.join(cmd)}")

        try:
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

            if result.returncode == 0:
                print(f"  ✅ 生成完成: {generated_dir}")
                return str(generated_dir)
            else:
                print(f"  ❌ 生成失败: {result.stderr}")
                return ""

        except Exception as e:
            print(f"  ❌ 生成异常: {e}")
            return ""

    def prepare_user_data_with_split(self, user_id: int, user_dir: str, other_dirs: list,
                                   max_samples_per_class: int = 500, negative_ratio: float = 2.0,
                                   train_ratio: float = 0.8) -> tuple:
        """
        为指定用户准备训练数据，支持训练/验证集划分

        Args:
            user_id: 用户ID
            user_dir: 该用户图像目录
            other_dirs: 其他用户目录列表
            max_samples_per_class: 正样本最大数量
            negative_ratio: 负样本与正样本的比例
            train_ratio: 训练集比例 (0.8 = 80%训练，20%验证)

        Returns:
            (train_paths, train_labels, val_paths, val_labels)
        """
        print(f"\n👤 准备用户 {user_id} 的数据 (训练/验证划分)")

        # 1. 收集正样本 (该用户的图像)
        user_path = Path(user_dir)
        positive_images = list(user_path.glob("*.png")) + list(user_path.glob("*.jpg"))
        positive_images = positive_images[:max_samples_per_class]

        print(f"  用户 {user_id} 总正样本: {len(positive_images)} 张")

        # 2. 划分正样本为训练/验证集
        random.shuffle(positive_images)
        train_split = int(len(positive_images) * train_ratio)

        train_positive = positive_images[:train_split]
        val_positive = positive_images[train_split:]

        print(f"  正样本划分: 训练 {len(train_positive)} 张, 验证 {len(val_positive)} 张")

        # 3. 收集负样本 (其他用户的图像)
        all_negative_images = []

        for other_dir in other_dirs:
            other_path = Path(other_dir)
            if other_path.exists():
                other_images = list(other_path.glob("*.png")) + list(other_path.glob("*.jpg"))
                all_negative_images.extend(other_images)

        # 4. 计算需要的负样本数量
        train_negative_needed = int(len(train_positive) * negative_ratio)
        val_negative_needed = int(len(val_positive) * negative_ratio)
        total_negative_needed = train_negative_needed + val_negative_needed

        print(f"  目标负样本: 训练 {train_negative_needed} 张, 验证 {val_negative_needed} 张")
        print(f"  可用负样本池: {len(all_negative_images)} 张")

        # 5. 随机选择负样本并划分
        if len(all_negative_images) >= total_negative_needed:
            selected_negative = random.sample(all_negative_images, total_negative_needed)
        else:
            selected_negative = all_negative_images
            print(f"  ⚠️  负样本不足，使用全部 {len(selected_negative)} 张")

        # 6. 划分负样本为训练/验证集
        random.shuffle(selected_negative)
        train_negative = selected_negative[:train_negative_needed]
        val_negative = selected_negative[train_negative_needed:train_negative_needed + val_negative_needed]

        print(f"  负样本划分: 训练 {len(train_negative)} 张, 验证 {len(val_negative)} 张")

        # 7. 组合训练集
        train_paths = [str(p) for p in train_positive] + [str(p) for p in train_negative]
        train_labels = [1] * len(train_positive) + [0] * len(train_negative)

        # 8. 组合验证集
        val_paths = [str(p) for p in val_positive] + [str(p) for p in val_negative]
        val_labels = [1] * len(val_positive) + [0] * len(val_negative)

        # 9. 打乱数据
        train_data = list(zip(train_paths, train_labels))
        val_data = list(zip(val_paths, val_labels))
        random.shuffle(train_data)
        random.shuffle(val_data)

        train_paths, train_labels = zip(*train_data) if train_data else ([], [])
        val_paths, val_labels = zip(*val_data) if val_data else ([], [])

        print(f"  ✅ 数据准备完成:")
        print(f"    训练集: {len(train_paths)} 张 (正样本 {len(train_positive)}, 负样本 {len(train_negative)})")
        print(f"    验证集: {len(val_paths)} 张 (正样本 {len(val_positive)}, 负样本 {len(val_negative)})")

        return list(train_paths), list(train_labels), list(val_paths), list(val_labels)
    
    def train_all_classifiers(self, epochs: int = 30, batch_size: int = 32, 
                            max_samples_per_class: int = 500) -> Dict[int, float]:
        """
        训练所有用户的分类器
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            max_samples_per_class: 每类最大样本数
            
        Returns:
            各用户分类器的最佳验证准确率
        """
        print(f"\n🤖 开始训练 {self.num_users} 个用户分类器")
        print(f"  参数: epochs={epochs}, batch_size={batch_size}, max_samples={max_samples_per_class}")
        print("=" * 60)
        
        classifier_accuracies = {}
        
        for user_id in sorted(self.user_mapping.keys()):
            print(f"\n👤 训练用户 {user_id} 的分类器...")
            
            # 准备该用户的训练数据 (支持训练/验证集划分)
            user_dir = self.user_mapping[user_id]
            other_dirs = [self.user_mapping[uid] for uid in self.user_mapping.keys() if uid != user_id]

            # 准备数据 (80%训练，20%验证)
            train_paths, train_labels, val_paths, val_labels = self.prepare_user_data_with_split(
                user_id=user_id,
                user_dir=user_dir,
                other_dirs=other_dirs,
                max_samples_per_class=max_samples_per_class,
                negative_ratio=2.0,  # 2:1的负正样本比例
                train_ratio=0.8      # 80%训练，20%验证
            )
            
            if len(train_paths) == 0:
                print(f"  ❌ 用户 {user_id} 无可用训练数据，跳过")
                continue

            # 训练分类器 (合并训练集和验证集，让原有方法内部划分)
            all_paths = train_paths + val_paths
            all_labels = train_labels + val_labels

            try:
                history = self.validation_system.train_user_classifier(
                    user_id=user_id,
                    image_paths=all_paths,
                    labels=all_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=5e-4,
                    validation_split=0.2  # 内部再次划分20%作为验证集
                )
                
                # 记录最佳准确率
                best_acc = max(history['val_acc'])
                classifier_accuracies[user_id] = best_acc
                
                # 保存分类器
                classifier_path = self.output_dir / f"classifier_user_{user_id:02d}.pth"
                self.validation_system.save_classifier(user_id, str(classifier_path))
                
                # 保存训练曲线
                plot_path = self.output_dir / f"training_user_{user_id:02d}.png"
                self.validation_system.plot_training_history(history, str(plot_path))
                
                print(f"  ✅ 用户 {user_id} 分类器训练完成，最佳准确率: {best_acc:.3f}")
                
            except Exception as e:
                print(f"  ❌ 用户 {user_id} 分类器训练失败: {e}")
                continue
        
        # 总结训练结果
        print(f"\n📊 分类器训练总结:")
        print(f"  成功训练: {len(classifier_accuracies)}/{self.num_users} 个分类器")
        
        if classifier_accuracies:
            avg_acc = np.mean(list(classifier_accuracies.values()))
            min_acc = min(classifier_accuracies.values())
            max_acc = max(classifier_accuracies.values())
            
            print(f"  平均准确率: {avg_acc:.3f}")
            print(f"  准确率范围: [{min_acc:.3f}, {max_acc:.3f}]")
            
            # 显示各用户准确率
            print(f"  详细结果:")
            for user_id in sorted(classifier_accuracies.keys()):
                acc = classifier_accuracies[user_id]
                status = "🌟" if acc >= 0.85 else "✅" if acc >= 0.75 else "⚠️" if acc >= 0.65 else "❌"
                print(f"    用户 {user_id:2d}: {acc:.3f} {status}")
        
        # 保存结果
        results_file = self.output_dir / "classifier_training_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'training_time': datetime.now().isoformat(),
                'num_users': self.num_users,
                'training_params': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'max_samples_per_class': max_samples_per_class
                },
                'classifier_accuracies': classifier_accuracies,
                'summary': {
                    'successful_classifiers': len(classifier_accuracies),
                    'average_accuracy': float(np.mean(list(classifier_accuracies.values()))) if classifier_accuracies else 0,
                    'min_accuracy': float(min(classifier_accuracies.values())) if classifier_accuracies else 0,
                    'max_accuracy': float(max(classifier_accuracies.values())) if classifier_accuracies else 0
                }
            }, f, indent=2)
        
        print(f"\n📄 训练结果保存在: {results_file}")
        
        return classifier_accuracies

    def load_classifiers(self, user_ids: Optional[List[int]] = None) -> bool:
        """
        加载已训练的分类器

        Args:
            user_ids: 要加载的用户ID列表，None表示加载所有可用的

        Returns:
            是否成功加载
        """
        if user_ids is None:
            user_ids = list(self.user_mapping.keys())

        loaded_count = 0

        for user_id in user_ids:
            classifier_path = self.output_dir / f"classifier_user_{user_id:02d}.pth"

            if classifier_path.exists():
                try:
                    self.validation_system.load_classifier(user_id, str(classifier_path))
                    loaded_count += 1
                    print(f"  ✅ 用户 {user_id} 分类器加载成功")
                except Exception as e:
                    print(f"  ❌ 用户 {user_id} 分类器加载失败: {e}")
            else:
                print(f"  ⚠️  用户 {user_id} 分类器文件不存在: {classifier_path}")

        print(f"📊 成功加载 {loaded_count}/{len(user_ids)} 个分类器")
        return loaded_count > 0

    def validate_generated_images(self, generated_images_dir: str, target_user_id: int,
                                confidence_threshold: float = 0.8) -> Dict:
        """
        验证生成图像是否包含指定用户特征

        Args:
            generated_images_dir: 生成图像目录
            target_user_id: 目标用户ID
            confidence_threshold: 置信度阈值

        Returns:
            验证结果
        """
        print(f"\n🔍 验证用户 {target_user_id} 的生成图像")
        print(f"  图像目录: {generated_images_dir}")
        print(f"  置信度阈值: {confidence_threshold}")

        # 检查目标用户分类器是否已加载
        if target_user_id not in self.validation_system.classifiers:
            print(f"❌ 用户 {target_user_id} 的分类器未加载")
            return {}

        # 验证生成图像
        try:
            result = self.validation_system.validate_generated_images(
                user_id=target_user_id,
                generated_images_dir=generated_images_dir,
                confidence_threshold=confidence_threshold
            )

            # 保存验证结果
            result_file = self.output_dir / f"validation_user_{target_user_id:02d}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"📄 验证结果保存在: {result_file}")

            return result

        except Exception as e:
            print(f"❌ 验证失败: {e}")
            return {}

    def cross_validate_all_users(self, generated_images_dir: str, target_user_id: int,
                               confidence_threshold: float = 0.8) -> Dict:
        """
        交叉验证：用所有用户的分类器验证生成图像
        这是验证条件扩散效果的关键测试

        Args:
            generated_images_dir: 生成图像目录
            target_user_id: 目标用户ID (生成图像声称的用户)
            confidence_threshold: 置信度阈值

        Returns:
            交叉验证结果
        """
        print(f"\n🎯 交叉验证：用所有分类器验证用户 {target_user_id} 的生成图像")
        print(f"  核心问题：生成的图像是否真的包含用户 {target_user_id} 的特征？")
        print("=" * 60)

        gen_dir = Path(generated_images_dir)
        if not gen_dir.exists():
            print(f"❌ 生成图像目录不存在: {gen_dir}")
            return {}

        image_files = list(gen_dir.glob("*.png")) + list(gen_dir.glob("*.jpg"))
        if not image_files:
            print(f"❌ 未找到生成图像")
            return {}

        print(f"📊 找到 {len(image_files)} 张生成图像")

        # 用每个用户的分类器验证这些图像
        cross_validation_results = {}

        for user_id in sorted(self.validation_system.classifiers.keys()):
            print(f"\n🔍 用户 {user_id} 的分类器验证...")

            try:
                result = self.validation_system.validate_generated_images(
                    user_id=user_id,
                    generated_images_dir=generated_images_dir,
                    confidence_threshold=confidence_threshold
                )

                cross_validation_results[user_id] = result

                success_rate = result['success_rate']
                avg_confidence = result['avg_confidence']

                if user_id == target_user_id:
                    # 目标用户的分类器应该识别出这些是该用户的图像
                    status = "🎯" if success_rate >= 0.7 else "⚠️" if success_rate >= 0.5 else "❌"
                    print(f"  {status} 目标用户 {user_id}: 成功率 {success_rate:.2f}, 置信度 {avg_confidence:.3f}")
                else:
                    # 其他用户的分类器应该识别出这些不是该用户的图像
                    status = "✅" if success_rate <= 0.3 else "⚠️" if success_rate <= 0.5 else "❌"
                    print(f"  {status} 其他用户 {user_id}: 成功率 {success_rate:.2f}, 置信度 {avg_confidence:.3f}")

            except Exception as e:
                print(f"  ❌ 用户 {user_id} 验证失败: {e}")
                continue

        # 分析交叉验证结果
        analysis = self._analyze_cross_validation(cross_validation_results, target_user_id, confidence_threshold)

        # 保存完整结果
        complete_result = {
            'target_user_id': target_user_id,
            'generated_images_dir': str(generated_images_dir),
            'confidence_threshold': confidence_threshold,
            'total_images': len(image_files),
            'cross_validation_results': cross_validation_results,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }

        result_file = self.output_dir / f"cross_validation_user_{target_user_id:02d}.json"
        with open(result_file, 'w') as f:
            json.dump(complete_result, f, indent=2)

        print(f"\n📄 交叉验证结果保存在: {result_file}")

        return complete_result

    def _analyze_cross_validation(self, cross_results: Dict, target_user_id: int,
                                confidence_threshold: float) -> Dict:
        """分析交叉验证结果"""

        if not cross_results:
            return {'error': 'No cross validation results'}

        # 目标用户结果
        target_result = cross_results.get(target_user_id, {})
        target_success_rate = target_result.get('success_rate', 0)
        target_confidence = target_result.get('avg_confidence', 0)

        # 其他用户结果
        other_results = {uid: result for uid, result in cross_results.items() if uid != target_user_id}
        other_success_rates = [result.get('success_rate', 0) for result in other_results.values()]
        other_confidences = [result.get('avg_confidence', 0) for result in other_results.values()]

        avg_other_success = np.mean(other_success_rates) if other_success_rates else 0
        avg_other_confidence = np.mean(other_confidences) if other_confidences else 0

        # 条件扩散效果评估
        condition_effective = (
            target_success_rate >= 0.7 and  # 目标用户分类器识别率高
            avg_other_success <= 0.3        # 其他用户分类器识别率低
        )

        # 计算区分度
        discrimination_score = target_success_rate - avg_other_success

        analysis = {
            'condition_effective': condition_effective,
            'target_user_performance': {
                'user_id': target_user_id,
                'success_rate': target_success_rate,
                'avg_confidence': target_confidence,
                'status': 'good' if target_success_rate >= 0.7 else 'poor'
            },
            'other_users_performance': {
                'avg_success_rate': avg_other_success,
                'avg_confidence': avg_other_confidence,
                'status': 'good' if avg_other_success <= 0.3 else 'poor'
            },
            'discrimination_score': discrimination_score,
            'overall_assessment': {
                'condition_control': 'effective' if condition_effective else 'ineffective',
                'discrimination_quality': (
                    'excellent' if discrimination_score >= 0.5 else
                    'good' if discrimination_score >= 0.3 else
                    'poor' if discrimination_score >= 0.1 else
                    'very_poor'
                )
            }
        }

        # 打印分析结果
        print(f"\n📊 交叉验证分析结果:")
        print(f"  目标用户 {target_user_id} 识别率: {target_success_rate:.2f}")
        print(f"  其他用户平均识别率: {avg_other_success:.2f}")
        print(f"  区分度得分: {discrimination_score:.2f}")
        print(f"  条件控制效果: {'✅ 有效' if condition_effective else '❌ 无效'}")

        return analysis

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="简化的条件扩散验证系统 - 专门验证31位用户步态微多普勒图像",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必需参数
    parser.add_argument("--data_dir", type=str, required=True,
                       help="数据目录 (包含ID_1, ID_2, ..., ID_31子目录)")
    parser.add_argument("--action", type=str, required=True,
                       choices=['train', 'validate', 'cross_validate'],
                       help="执行动作: train=训练分类器, validate=验证单个用户, cross_validate=交叉验证")

    # 可选参数
    parser.add_argument("--output_dir", type=str, default="./validation_results",
                       help="输出目录")
    parser.add_argument("--device", type=str, default="auto",
                       help="计算设备 (auto/cuda/cpu)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=30,
                       help="分类器训练轮数")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批次大小")
    parser.add_argument("--max_samples", type=int, default=500,
                       help="每类最大样本数")

    # 验证参数
    parser.add_argument("--generated_images_dir", type=str,
                       help="生成图像目录 (验证时必需)")
    parser.add_argument("--target_user_id", type=int,
                       help="目标用户ID (验证时必需)")
    parser.add_argument("--confidence_threshold", type=float, default=0.8,
                       help="置信度阈值")

    args = parser.parse_args()

    print("🎯 简化的条件扩散验证系统")
    print("=" * 60)
    print(f"📊 数据目录: {args.data_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🎬 执行动作: {args.action}")

    # 创建验证器
    validator = SimpleConditionValidator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )

    if args.action == 'train':
        # 训练所有用户的分类器
        print(f"\n🤖 开始训练分类器...")
        accuracies = validator.train_all_classifiers(
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_samples_per_class=args.max_samples
        )

        if accuracies:
            print(f"\n🎉 分类器训练完成!")
            print(f"💡 下一步: 使用 --action validate 或 --action cross_validate 验证生成图像")
        else:
            print(f"\n❌ 分类器训练失败!")
            return 1

    elif args.action == 'validate':
        # 验证单个用户的生成图像
        if not args.generated_images_dir or args.target_user_id is None:
            print(f"❌ 验证模式需要 --generated_images_dir 和 --target_user_id 参数")
            return 1

        # 加载分类器
        if not validator.load_classifiers([args.target_user_id]):
            print(f"❌ 无法加载用户 {args.target_user_id} 的分类器")
            return 1

        # 验证
        result = validator.validate_generated_images(
            generated_images_dir=args.generated_images_dir,
            target_user_id=args.target_user_id,
            confidence_threshold=args.confidence_threshold
        )

        if result:
            success_rate = result['success_rate']
            avg_confidence = result['avg_confidence']
            print(f"\n📊 验证结果: 成功率 {success_rate:.2f}, 平均置信度 {avg_confidence:.3f}")

            if success_rate >= 0.7:
                print(f"🎉 验证成功! 生成图像包含用户 {args.target_user_id} 的特征")
            else:
                print(f"⚠️  验证结果不理想，可能需要改进条件扩散模型")
        else:
            print(f"❌ 验证失败!")
            return 1

    elif args.action == 'cross_validate':
        # 交叉验证
        if not args.generated_images_dir or args.target_user_id is None:
            print(f"❌ 交叉验证模式需要 --generated_images_dir 和 --target_user_id 参数")
            return 1

        # 加载所有分类器
        if not validator.load_classifiers():
            print(f"❌ 无法加载分类器")
            return 1

        # 交叉验证
        result = validator.cross_validate_all_users(
            generated_images_dir=args.generated_images_dir,
            target_user_id=args.target_user_id,
            confidence_threshold=args.confidence_threshold
        )

        if result and 'analysis' in result:
            analysis = result['analysis']
            condition_effective = analysis['condition_effective']
            discrimination_score = analysis['discrimination_score']

            if condition_effective:
                print(f"\n🎉 交叉验证成功! 条件扩散模型有效控制用户特征")
                print(f"  区分度得分: {discrimination_score:.2f}")
            else:
                print(f"\n⚠️  交叉验证显示条件控制效果不佳")
                print(f"  区分度得分: {discrimination_score:.2f}")
                print(f"💡 建议: 检查训练数据、增加训练轮数或调整模型架构")
        else:
            print(f"❌ 交叉验证失败!")
            return 1

    return 0

if __name__ == "__main__":
    exit(main())
