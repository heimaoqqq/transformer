#!/usr/bin/env python3
"""
基于度量学习的验证器 - 解决小数据量问题
使用Siamese Network学习用户相似性，而不是训练独立分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict
import random
from tqdm import tqdm

class SiameseDataset(Dataset):
    """Siamese网络数据集 - 生成图像对"""
    
    def __init__(self, user_images_dict: Dict[int, List[str]], transform=None):
        """
        Args:
            user_images_dict: {user_id: [image_paths]}
            transform: 图像变换
        """
        self.user_images = user_images_dict
        self.user_ids = list(user_images_dict.keys())
        self.transform = transform
        
        # 生成所有可能的图像对
        self.pairs = []
        self.labels = []
        
        # 正样本对：同一用户的不同图像
        for user_id, images in user_images_dict.items():
            if len(images) >= 2:
                for i in range(len(images)):
                    for j in range(i+1, min(i+10, len(images))):  # 限制正样本对数量
                        self.pairs.append((images[i], images[j]))
                        self.labels.append(1)  # 同一用户
        
        # 负样本对：不同用户的图像
        num_negative = len(self.pairs)  # 与正样本对数量相等
        for _ in range(num_negative):
            user1, user2 = random.sample(self.user_ids, 2)
            img1 = random.choice(user_images_dict[user1])
            img2 = random.choice(user_images_dict[user2])
            self.pairs.append((img1, img2))
            self.labels.append(0)  # 不同用户
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        
        # 加载图像
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label

class ImprovedSiameseNetwork(nn.Module):
    """改进的Siamese网络 - 专门处理相似特征的小数据问题"""

    def __init__(self, embedding_dim=256):
        super(ImprovedSiameseNetwork, self).__init__()

        # 使用更深的网络提取细微特征
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=True)

        # 移除最后的分类层
        self.backbone.fc = nn.Identity()

        # 多尺度特征提取
        self.multi_scale = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局特征
            nn.AdaptiveAvgPool2d((2, 2)),  # 中等尺度
            nn.AdaptiveAvgPool2d((4, 4)),  # 细粒度特征
        ])

        # 注意力机制 - 聚焦关键特征
        self.attention = nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)

        # 特征融合和降维
        self.feature_fusion = nn.Sequential(
            nn.Linear(2048 * (1 + 4 + 16), 1024),  # 多尺度特征融合
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 学习如何比较特征（而不是简单的余弦相似度）
        self.relation_module = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def extract_deep_features(self, x):
        """提取深层多尺度特征"""
        # 通过backbone提取基础特征
        features = self.backbone(x)  # [batch, 2048, 7, 7]

        # 多尺度特征提取
        multi_scale_features = []
        for pool in self.multi_scale:
            pooled = pool(features)  # [batch, 2048, scale, scale]
            flattened = pooled.view(pooled.size(0), -1)  # [batch, 2048*scale*scale]
            multi_scale_features.append(flattened)

        # 拼接多尺度特征
        combined_features = torch.cat(multi_scale_features, dim=1)  # [batch, 2048*(1+4+16)]

        # 特征融合
        fused_features = self.feature_fusion(combined_features)  # [batch, embedding_dim]

        return fused_features

    def forward_one(self, x):
        """单个图像的前向传播"""
        return self.extract_deep_features(x)

    def forward(self, img1, img2):
        """计算两个图像的相似性 - 使用学习的关系模块"""
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)

        # L2归一化
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        # 使用关系模块学习如何比较特征
        combined = torch.cat([emb1, emb2], dim=1)
        similarity = self.relation_module(combined).squeeze()

        return similarity, emb1, emb2

class MetricLearningValidator:
    """基于度量学习的验证器"""
    
    def __init__(self, device="auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🚀 使用设备: {self.device}")
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = None
        self.user_prototypes = {}  # 存储每个用户的原型特征
    
    def load_user_images(self, data_root: str) -> Dict[int, List[str]]:
        """加载所有用户的图像路径"""
        data_path = Path(data_root)
        user_images = {}
        
        for user_dir in data_path.iterdir():
            if user_dir.is_dir() and user_dir.name.startswith('ID_'):
                try:
                    user_id = int(user_dir.name.split('_')[1])
                    images = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))
                    if images:
                        user_images[user_id] = [str(p) for p in images]
                        print(f"  用户 {user_id}: {len(images)} 张图像")
                except ValueError:
                    continue
        
        return user_images
    
    def train_siamese_network(self, user_images: Dict[int, List[str]], 
                            epochs: int = 50, batch_size: int = 32) -> Dict:
        """训练Siamese网络"""
        print(f"\n🎯 训练Siamese网络...")
        
        # 创建数据集
        dataset = SiameseDataset(user_images, self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        print(f"  📊 数据集大小: {len(dataset)} 个图像对")
        print(f"  📊 正样本对: {sum(dataset.labels)} 个")
        print(f"  📊 负样本对: {len(dataset.labels) - sum(dataset.labels)} 个")
        
        # 创建改进的模型
        self.model = ImprovedSiameseNetwork().to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        # 训练循环
        history = {'train_loss': [], 'train_acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for img1, img2, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.float().to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                similarity, _, _ = self.model(img1, img2)
                
                # 损失计算
                loss = criterion(similarity, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 准确率计算
                predicted = (torch.sigmoid(similarity) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(accuracy)
            
            print(f"  Epoch {epoch+1}: Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
        
        print(f"✅ Siamese网络训练完成")
        return history
    
    def compute_user_prototypes(self, user_images: Dict[int, List[str]]):
        """计算每个用户的原型特征"""
        print(f"\n📊 计算用户原型特征...")
        
        self.model.eval()
        self.user_prototypes = {}
        
        with torch.no_grad():
            for user_id, image_paths in user_images.items():
                embeddings = []
                
                for img_path in image_paths:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    embedding = self.model.forward_one(img_tensor)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    embeddings.append(embedding.cpu())
                
                # 计算原型（平均特征）
                prototype = torch.stack(embeddings).mean(dim=0)
                self.user_prototypes[user_id] = prototype
                
                print(f"  用户 {user_id}: 原型特征计算完成 ({len(embeddings)} 张图像)")
    
    def validate_generated_images(self, target_user_id: int, generated_images_dir: str,
                                threshold: float = 0.5) -> Dict:
        """验证生成图像是否属于目标用户"""
        print(f"\n🔍 验证生成图像 (度量学习方法)")
        
        if target_user_id not in self.user_prototypes:
            raise ValueError(f"用户 {target_user_id} 的原型特征不存在")
        
        # 加载生成图像
        gen_dir = Path(generated_images_dir)
        image_files = list(gen_dir.glob("*.png")) + list(gen_dir.glob("*.jpg"))
        
        if not image_files:
            return {'error': 'No generated images found'}
        
        print(f"  找到 {len(image_files)} 张生成图像")
        
        # 获取目标用户原型
        target_prototype = self.user_prototypes[target_user_id].to(self.device)
        
        # 计算相似性
        similarities = []
        self.model.eval()
        
        with torch.no_grad():
            for img_path in image_files:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                # 提取特征
                embedding = self.model.forward_one(img_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)
                
                # 计算与目标用户原型的相似性
                similarity = F.cosine_similarity(embedding, target_prototype, dim=1)
                similarities.append(similarity.item())
        
        # 统计结果
        similarities = np.array(similarities)
        success_count = (similarities > threshold).sum()
        success_rate = success_count / len(similarities)
        avg_similarity = similarities.mean()
        
        result = {
            'success_rate': success_rate,
            'avg_similarity': avg_similarity,
            'threshold': threshold,
            'total_images': len(similarities),
            'successful_images': int(success_count),
            'similarities': similarities.tolist()
        }
        
        print(f"  📊 验证结果:")
        print(f"    成功率: {success_rate:.3f}")
        print(f"    平均相似性: {avg_similarity:.3f}")
        print(f"    阈值: {threshold}")
        
        return result

# 使用示例
if __name__ == "__main__":
    validator = MetricLearningValidator()
    
    # 加载数据
    user_images = validator.load_user_images("data/processed")
    
    # 训练Siamese网络
    history = validator.train_siamese_network(user_images, epochs=30)
    
    # 计算用户原型
    validator.compute_user_prototypes(user_images)
    
    # 验证生成图像
    result = validator.validate_generated_images(
        target_user_id=1,
        generated_images_dir="validation_results/generated_images",
        threshold=0.7
    )
