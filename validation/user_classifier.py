#!/usr/bin/env python3
"""
用户验证分类器
使用ResNet-18为每个用户训练独立的二分类器
验证生成图像是否包含对应用户的特征信息
"""

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
from typing import List, Tuple, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

class UserImageDataset(Dataset):
    """用户图像数据集"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Args:
            image_paths: 图像路径列表
            labels: 标签列表 (0: 负样本, 1: 正样本)
            transform: 图像变换
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        assert len(image_paths) == len(labels), "图像数量与标签数量不匹配"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {image_path}: {e}")
            # 创建一个黑色图像作为备用
            image = Image.new('RGB', (64, 64), (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

class UserClassifier(nn.Module):
    """标准的用户分类器 - 使用成熟的ResNet架构"""

    def __init__(self, num_classes=2, pretrained=True):
        """
        Args:
            num_classes: 分类数量 (2: 是/不是该用户)
            pretrained: 是否使用预训练权重
        """
        super(UserClassifier, self).__init__()

        # 使用标准的ResNet-18
        self.backbone = resnet18(pretrained=pretrained)

        # 替换最后的分类层为标准做法
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # 标准的ResNet前向传播
        return self.backbone(x)

class MicroDopplerCNN(nn.Module):
    """专门为微多普勒时频图设计的轻量级CNN"""

    def __init__(self, num_classes=2, dropout_rate=0.5):
        """
        专门针对微多普勒时频图的特征设计
        关注时间-频率域的局部模式
        """
        super(MicroDopplerCNN, self).__init__()

        # 特征提取层 - 专门捕获时频图特征
        self.features = nn.Sequential(
            # 第一组：捕获粗粒度时频特征
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 第二组：捕获中等粒度特征
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三组：捕获细粒度特征
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四组：高级特征
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # 自适应池化到固定尺寸
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(128, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 特征提取
        features = self.features(x)  # [batch_size, 256, 4, 4]

        # 展平
        features = features.view(features.size(0), -1)  # [batch_size, 256*4*4]

        # 分类
        output = self.classifier(features)

        return output

class UserValidationSystem:
    """用户验证系统"""
    
    def __init__(self, device="auto"):
        """
        Args:
            device: 计算设备
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🚀 使用设备: {self.device}")
        
        # 标准的图像预处理 - 不使用数据增强
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),  # 标准ImageNet尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet标准化
        ])

        # 验证时使用相同的变换
        self.val_transform = self.transform
        
        # 存储训练好的分类器
        self.classifiers = {}
    
    def prepare_user_data(self, user_id: int, real_images_dir: str, other_users_dirs: List[str],
                         max_samples_per_class: int = 500, negative_ratio: float = 3.0) -> Tuple[List[str], List[int]]:
        """
        为指定用户准备训练数据

        Args:
            user_id: 用户ID
            real_images_dir: 该用户真实图像目录
            other_users_dirs: 其他用户图像目录列表
            max_samples_per_class: 正样本最大数量
            negative_ratio: 负样本与正样本的比例 (默认3:1)

        Returns:
            (image_paths, labels): 图像路径列表和标签列表
        """
        image_paths = []
        labels = []
        
        # 正样本: 该用户的真实图像
        real_dir = Path(real_images_dir)
        if real_dir.exists():
            real_images = list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg"))
            real_images = real_images[:max_samples_per_class]  # 限制样本数量
            
            image_paths.extend([str(p) for p in real_images])
            labels.extend([1] * len(real_images))
            
            print(f"  用户 {user_id} 正样本: {len(real_images)} 张")
        else:
            print(f"  警告: 用户 {user_id} 真实图像目录不存在: {real_dir}")
        
        # 负样本: 其他用户的图像 (改进的采样策略)
        max_negative_samples = int(len(real_images) * negative_ratio)
        print(f"  目标负样本数量: {max_negative_samples} 张 (比例 {negative_ratio}:1)")

        # 收集所有其他用户的图像
        all_negative_images = []
        for other_dir in other_users_dirs:
            other_path = Path(other_dir)
            if other_path.exists():
                other_images = list(other_path.glob("*.png")) + list(other_path.glob("*.jpg"))
                all_negative_images.extend(other_images)

        # 随机采样负样本，确保代表性
        import random
        if len(all_negative_images) > max_negative_samples:
            selected_negative = random.sample(all_negative_images, max_negative_samples)
        else:
            selected_negative = all_negative_images
            print(f"  警告: 可用负样本({len(all_negative_images)})少于目标数量({max_negative_samples})")

        # 添加负样本
        image_paths.extend([str(p) for p in selected_negative])
        labels.extend([0] * len(selected_negative))
        negative_count = len(selected_negative)
        
        print(f"  用户 {user_id} 负样本: {negative_count} 张")
        print(f"  用户 {user_id} 总样本: {len(image_paths)} 张")
        
        return image_paths, labels
    
    def train_user_classifier(self, user_id: int, image_paths: List[str], labels: List[int],
                            epochs: int = 20, batch_size: int = 32, learning_rate: float = 1e-3,
                            validation_split: float = 0.2) -> Dict:
        """
        训练用户分类器
        
        Args:
            user_id: 用户ID
            image_paths: 图像路径列表
            labels: 标签列表
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            validation_split: 验证集比例
            
        Returns:
            训练历史字典
        """
        print(f"\n🎯 训练用户 {user_id} 的分类器...")
        
        # 划分训练集和验证集
        total_samples = len(image_paths)
        val_size = int(total_samples * validation_split)
        train_size = total_samples - val_size
        
        # 随机打乱数据
        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_paths = [image_paths[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_paths = [image_paths[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        # 创建数据集和数据加载器
        train_dataset = UserImageDataset(train_paths, train_labels, self.transform)
        val_dataset = UserImageDataset(val_paths, val_labels, self.transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # 使用标准的ResNet-18分类器
        model = UserClassifier(num_classes=2, pretrained=True)
        print(f"  🏗️  使用标准ResNet-18分类器")

        model.to(self.device)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  📊 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        
        # 标准的损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # 使用余弦退火调度器（更现代的调度策略）
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        best_model_state = None

        # 早停机制
        patience = 10  # 10个epoch没有改善就停止
        patience_counter = 0
        min_improvement = 0.001  # 最小改善阈值，避免微小波动触发早停
        
        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels_batch in val_loader:
                    images = images.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels_batch)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels_batch.size(0)
                    val_correct += (predicted == labels_batch).sum().item()
            
            # 计算平均指标
            train_loss_avg = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            val_loss_avg = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            # 保存历史
            history['train_loss'].append(train_loss_avg)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss_avg)
            history['val_acc'].append(val_acc)
            
            # 保存最佳模型
            # 早停和最佳模型保存（修复逻辑）
            if val_acc > best_val_acc + min_improvement:
                # 显著改善：更新最佳模型并重置计数器
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f"    ✅ 验证准确率改善: {val_acc:.4f} (最佳: {best_val_acc:.4f})")
            else:
                # 没有显著改善：增加计数器
                patience_counter += 1
                print(f"    ⏳ 无改善计数: {patience_counter}/{patience}")

            # 更新学习率
            scheduler.step()

            print(f"  Epoch {epoch+1}: Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Val Loss: {val_loss_avg:.4f}")

            # 早停检查
            if patience_counter >= patience:
                print(f"  🛑 早停触发：连续{patience}个epoch验证准确率无显著改善，停止训练")
                print(f"  📊 最终最佳验证准确率: {best_val_acc:.4f}")
                break
        
        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"  📥 已加载最佳模型状态")
        else:
            print(f"  ⚠️  未找到更好的模型，使用最终状态")

        # 保存分类器
        self.classifiers[user_id] = model

        # 训练完成总结
        final_epoch = len(history['val_acc'])
        print(f"✅ 用户 {user_id} 分类器训练完成")
        print(f"  📊 训练轮数: {final_epoch}/{epochs}")
        print(f"  🎯 最佳验证准确率: {best_val_acc:.4f}")
        print(f"  📈 最终训练准确率: {history['train_acc'][-1]:.4f}")

        # 判断训练质量
        if best_val_acc >= 0.85:
            print(f"  🌟 训练质量: 优秀")
        elif best_val_acc >= 0.75:
            print(f"  ✅ 训练质量: 良好")
        elif best_val_acc >= 0.65:
            print(f"  ⚠️  训练质量: 一般")
        else:
            print(f"  ❌ 训练质量: 较差，建议检查数据或增加训练轮数")

        return history

    def validate_generated_images(self, user_id: int, generated_images_dir: str,
                                confidence_threshold: float = 0.8) -> Dict:
        """
        验证生成图像是否包含用户特征

        Args:
            user_id: 用户ID
            generated_images_dir: 生成图像目录
            confidence_threshold: 置信度阈值 (>0.8算成功)

        Returns:
            验证结果字典
        """
        if user_id not in self.classifiers:
            raise ValueError(f"用户 {user_id} 的分类器尚未训练")

        print(f"\n🔍 验证用户 {user_id} 的生成图像...")

        # 加载生成图像
        gen_dir = Path(generated_images_dir)
        if not gen_dir.exists():
            raise FileNotFoundError(f"生成图像目录不存在: {gen_dir}")

        image_files = list(gen_dir.glob("*.png")) + list(gen_dir.glob("*.jpg"))
        if not image_files:
            print(f"  警告: 未找到生成图像")
            return {}

        print(f"  找到 {len(image_files)} 张生成图像")

        # 准备数据
        image_paths = [str(p) for p in image_files]
        dataset = UserImageDataset(image_paths, [0] * len(image_paths), self.transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

        # 获取分类器
        model = self.classifiers[user_id]
        model.eval()

        # 预测
        all_predictions = []
        all_confidences = []

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="验证生成图像"):
                images = images.to(self.device)
                outputs = model(images)

                # 计算概率和预测
                probabilities = torch.softmax(outputs, dim=1)
                confidences = probabilities[:, 1]  # 正类(该用户)的置信度
                predictions = (confidences > confidence_threshold).int()

                all_predictions.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

        # 计算统计信息
        all_predictions = np.array(all_predictions)
        all_confidences = np.array(all_confidences)

        success_count = np.sum(all_predictions)
        success_rate = success_count / len(all_predictions)
        avg_confidence = np.mean(all_confidences)
        max_confidence = np.max(all_confidences)
        min_confidence = np.min(all_confidences)

        results = {
            'user_id': user_id,
            'total_images': len(image_files),
            'success_count': int(success_count),
            'success_rate': float(success_rate),
            'avg_confidence': float(avg_confidence),
            'max_confidence': float(max_confidence),
            'min_confidence': float(min_confidence),
            'confidence_threshold': confidence_threshold,
            'individual_confidences': all_confidences.tolist(),
            'individual_predictions': all_predictions.tolist(),
            'image_files': [p.name for p in image_files]
        }

        print(f"  验证结果:")
        print(f"    成功图像: {success_count}/{len(image_files)} ({success_rate:.1%})")
        print(f"    平均置信度: {avg_confidence:.3f}")
        print(f"    置信度范围: [{min_confidence:.3f}, {max_confidence:.3f}]")

        return results

    def save_classifier(self, user_id: int, save_path: str):
        """保存分类器"""
        if user_id not in self.classifiers:
            raise ValueError(f"用户 {user_id} 的分类器尚未训练")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.classifiers[user_id].state_dict(),
            'user_id': user_id,
            'model_class': 'UserClassifier'
        }, save_path)

        print(f"✅ 用户 {user_id} 分类器已保存到: {save_path}")

    def load_classifier(self, user_id: int, load_path: str):
        """加载分类器"""
        checkpoint = torch.load(load_path, map_location=self.device)

        model = UserClassifier(num_classes=2, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        self.classifiers[user_id] = model

        print(f"✅ 用户 {user_id} 分类器已从 {load_path} 加载")

    def plot_training_history(self, history: Dict, save_path: str = None):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # 准确率曲线
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 训练历史图已保存到: {save_path}")

        plt.show()

    def generate_validation_report(self, results_list: List[Dict], save_path: str = None) -> str:
        """生成验证报告"""
        report = []
        report.append("# 用户生成图像验证报告\n")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"置信度阈值: {results_list[0]['confidence_threshold']}\n\n")

        # 总体统计
        total_images = sum(r['total_images'] for r in results_list)
        total_success = sum(r['success_count'] for r in results_list)
        overall_success_rate = total_success / total_images if total_images > 0 else 0

        report.append("## 总体统计\n")
        report.append(f"- 总图像数: {total_images}\n")
        report.append(f"- 成功图像数: {total_success}\n")
        report.append(f"- 总体成功率: {overall_success_rate:.1%}\n\n")

        # 各用户详细结果
        report.append("## 各用户详细结果\n")
        for result in results_list:
            user_id = result['user_id']
            report.append(f"### 用户 {user_id}\n")
            report.append(f"- 图像数量: {result['total_images']}\n")
            report.append(f"- 成功数量: {result['success_count']}\n")
            report.append(f"- 成功率: {result['success_rate']:.1%}\n")
            report.append(f"- 平均置信度: {result['avg_confidence']:.3f}\n")
            report.append(f"- 置信度范围: [{result['min_confidence']:.3f}, {result['max_confidence']:.3f}]\n\n")

        report_text = "".join(report)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 验证报告已保存到: {save_path}")

        return report_text
