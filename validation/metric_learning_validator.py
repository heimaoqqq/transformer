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

class SmallDataSiameseNetwork(nn.Module):
    """小数据集专用Siamese网络 - 防止过拟合"""

    def __init__(self, embedding_dim=64):
        super(SmallDataSiameseNetwork, self).__init__()

        # 使用ResNet18，参数量适中
        self.backbone = resnet18(pretrained=True)
        print("  🎯 小数据集优化：使用ResNet18 + 简化分类头")

        # 简化的特征提取头（防止过拟合）
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),  # 更强的dropout
            nn.Linear(in_features, embedding_dim),  # 直接降到64维
        )

        # 相似性计算
        self.similarity = nn.CosineSimilarity(dim=1)

        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  📊 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        
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
            if torch.cuda.is_available():
                try:
                    # 测试CUDA是否正常工作
                    test_tensor = torch.randn(10, 10).cuda()
                    _ = test_tensor + test_tensor
                    self.device = torch.device("cuda")
                    print(f"🚀 使用设备: {self.device}")
                except Exception as e:
                    print(f"⚠️ CUDA测试失败: {e}")
                    print(f"🔄 回退到CPU训练")
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
                print(f"🚀 使用设备: {self.device}")
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
        
        # 创建数据集（优化数据加载）
        dataset = SiameseDataset(user_images, self.transform)

        # 小数据集自动调整批次大小
        if len(dataset) < 1000 and batch_size > 16:
            recommended_batch_size = max(8, len(dataset) // 50)  # 确保至少50个batch
            print(f"  🔧 小数据集自动调整: batch_size {batch_size} -> {recommended_batch_size}")
            batch_size = recommended_batch_size

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # P100环境减少worker数量
            pin_memory=True,  # 加速GPU传输
            persistent_workers=False  # P100环境关闭持久worker
        )
        
        print(f"  📊 数据集大小: {len(dataset)} 个图像对")
        print(f"  📊 正样本对: {sum(dataset.labels)} 个")
        print(f"  📊 负样本对: {len(dataset.labels) - sum(dataset.labels)} 个")

        # 小数据集批次大小优化
        total_batches = len(dataloader)
        if total_batches < 50:
            print(f"  ⚠️  小数据集警告: 每epoch只有{total_batches}个batch")
            print(f"  💡 建议: 考虑减小batch_size以增加梯度更新频率")

        print(f"  🔧 训练配置: batch_size={batch_size}, 每epoch {total_batches} 个batch")
        
        # 创建小数据集专用模型（防止过拟合）
        self.model = SmallDataSiameseNetwork(embedding_dim=64).to(self.device)
        
        # 小数据集优化配置（防止过拟合）
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,  # 降低学习率，防止过拟合
            weight_decay=1e-2,  # 增强权重衰减
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )

        criterion = nn.MSELoss()  # 使用MSE损失，因为相似度在[-1,1]范围

        # P100不使用混合精度（Pascal架构支持有限）
        print(f"  🔧 P100优化：不使用混合精度训练")

        # 早停机制（防止过拟合）
        best_loss = float('inf')
        patience = 5  # 小数据集用更小的patience
        patience_counter = 0

        # 训练循环
        history = {'train_loss': [], 'train_acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for img1, img2, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                img1, img2, labels = img1.to(self.device, non_blocking=True), img2.to(self.device, non_blocking=True), labels.float().to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # 标准前向传播（P100优化）
                similarity, _, _ = self.model(img1, img2)
                target_similarity = labels * 2.0 - 1.0  # [0,1] -> [-1,1]
                loss = criterion(similarity, target_similarity)

                # 标准反向传播
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # 准确率计算：相似度>0认为是同一用户
                predicted = (similarity > 0).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            # 更新学习率
            scheduler.step()

            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total

            history['train_loss'].append(avg_loss)
            history['train_acc'].append(accuracy)

            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                print(f"  Epoch {epoch+1}: Loss: {avg_loss:.4f}, Acc: {accuracy:.4f} ✅")
            else:
                patience_counter += 1
                print(f"  Epoch {epoch+1}: Loss: {avg_loss:.4f}, Acc: {accuracy:.4f} (无改善: {patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"  🛑 早停触发：{patience}个epoch无改善，防止过拟合")
                    break
        
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
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="相似性阈值（针对高相似性数据降低）")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批次大小（小数据集推荐16-32）")
    parser.add_argument("--use_larger_model", action="store_true",
                       help="使用更大的模型（ResNet34）")
    parser.add_argument("--force_cpu", action="store_true",
                       help="强制使用CPU训练（GPU有问题时）")

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

    # 训练Siamese网络（使用优化参数）
    history = validator.train_siamese_network(
        user_images,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

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
