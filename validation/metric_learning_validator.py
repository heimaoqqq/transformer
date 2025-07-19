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

class SimplifiedSiameseNetwork(nn.Module):
    """简化的Siamese网络 - 稳定可靠的实现"""

    def __init__(self, embedding_dim=128):
        super(SimplifiedSiameseNetwork, self).__init__()

        # 使用ResNet18，更稳定
        self.backbone = resnet18(pretrained=True)

        # 替换最后的分类层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 相似性计算
        self.similarity = nn.CosineSimilarity(dim=1)
        
    def forward_one(self, x):
        """单个图像的前向传播"""
        return self.backbone(x)

    def forward(self, img1, img2):
        """计算两个图像的相似性"""
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)

        # L2归一化
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        # 余弦相似性
        similarity = self.similarity(emb1, emb2)

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
        
        # 创建简化的模型
        self.model = SimplifiedSiameseNetwork().to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()  # 使用MSE损失，因为相似度在[-1,1]范围
        
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

                # 将标签转换为相似度目标：1->1.0, 0->-1.0
                target_similarity = labels * 2.0 - 1.0  # [0,1] -> [-1,1]

                # 损失计算
                loss = criterion(similarity, target_similarity)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # 准确率计算：相似度>0认为是同一用户
                predicted = (similarity > 0).float()
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
    import argparse

    parser = argparse.ArgumentParser(description="度量学习验证器 - 不需要预训练模型")
    parser.add_argument("--data_root", type=str, default="/kaggle/input/dataset",
                       help="真实数据目录路径")
    parser.add_argument("--target_user_id", type=int, default=1,
                       help="目标用户ID")
    parser.add_argument("--generated_images_dir", type=str,
                       default="/kaggle/working/validation_results/generated_images",
                       help="生成图像目录路径")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Siamese网络训练轮数")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="相似性阈值")

    args = parser.parse_args()

    print("🧠 度量学习验证器 - 针对相似特征优化")
    print(f"🔧 配置:")
    print(f"  数据目录: {args.data_root}")
    print(f"  目标用户: {args.target_user_id}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  相似性阈值: {args.threshold}")

    validator = MetricLearningValidator()

    # 加载数据
    user_images = validator.load_user_images(args.data_root)

    if not user_images:
        print("❌ 未找到用户数据，请检查数据目录路径")
        exit(1)

    # 训练Siamese网络
    history = validator.train_siamese_network(user_images, epochs=args.epochs)

    # 计算用户原型
    validator.compute_user_prototypes(user_images)

    # 验证生成图像（如果存在）
    from pathlib import Path
    if Path(args.generated_images_dir).exists():
        result = validator.validate_generated_images(
            target_user_id=args.target_user_id,
            generated_images_dir=args.generated_images_dir,
            threshold=args.threshold
        )
        print(f"\n✅ 度量学习验证完成")
    else:
        print(f"\n📋 Siamese网络训练完成，生成图像目录不存在，跳过验证步骤")
        print(f"💡 提示: 先运行传统验证器生成图像，再运行度量学习验证")
