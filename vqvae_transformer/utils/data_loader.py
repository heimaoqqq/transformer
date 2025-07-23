#!/usr/bin/env python3
"""
数据加载器
复用主项目的数据加载逻辑，适配VQ-VAE + Transformer需求
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict
import random
import numpy as np

# 添加主项目路径以复用数据加载器
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from utils.data_loader import MicroDopplerDataset as BaseMicroDopplerDataset
except ImportError:
    # 如果无法导入，提供基础实现
    class BaseMicroDopplerDataset(Dataset):
        def __init__(self, data_dir, transform=None, return_user_id=False):
            self.data_dir = Path(data_dir)
            self.transform = transform
            self.return_user_id = return_user_id
            
            # 加载数据
            self.data = []
            self._load_data()
        
        def _load_data(self):
            """加载数据"""
            for user_dir in self.data_dir.iterdir():
                if user_dir.is_dir() and user_dir.name.startswith('ID'):
                    try:
                        # 处理 ID1, ID_2, ID_3 等格式
                        dir_name = user_dir.name
                        if '_' in dir_name:
                            user_id = int(dir_name.split('_')[1])  # ID_2 -> 2
                        else:
                            user_id = int(dir_name[2:])  # ID1 -> 1

                        # 收集所有图像文件
                        for ext in ['*.png', '*.jpg', '*.jpeg']:
                            for img_path in user_dir.glob(ext):
                                self.data.append({
                                    'image_path': str(img_path),
                                    'user_id': user_id,
                                })
                    except ValueError:
                        continue
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]

            # 加载图像
            image = Image.open(item['image_path']).convert('RGB')

            if self.transform:
                image = self.transform(image)

            if self.return_user_id:
                return {
                    'image': image,
                    'user_id': item['user_id']
                }
            else:
                return image

class MicroDopplerDataset(BaseMicroDopplerDataset):
    """
    微多普勒数据集
    扩展基础数据集以支持VQ-VAE + Transformer的需求
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        return_user_id: bool = False,
        user_filter: Optional[List[int]] = None,
        max_samples_per_user: Optional[int] = None,
    ):
        """
        Args:
            data_dir: 数据目录路径
            transform: 图像变换
            return_user_id: 是否返回用户ID
            user_filter: 用户ID过滤列表
            max_samples_per_user: 每个用户的最大样本数
        """
        self.user_filter = user_filter
        self.max_samples_per_user = max_samples_per_user
        
        super().__init__(data_dir, transform, return_user_id)
        
        # 应用过滤器
        if self.user_filter is not None:
            self.data = [item for item in self.data if item['user_id'] in self.user_filter]
        
        # 限制每个用户的样本数
        if self.max_samples_per_user is not None:
            self._limit_samples_per_user()
        
        print(f"📊 数据集加载完成:")
        print(f"   总样本数: {len(self.data)}")
        print(f"   用户数量: {len(set(item['user_id'] for item in self.data))}")
    
    def _limit_samples_per_user(self):
        """限制每个用户的样本数"""
        user_counts = {}
        filtered_data = []
        
        for item in self.data:
            user_id = item['user_id']
            if user_id not in user_counts:
                user_counts[user_id] = 0
            
            if user_counts[user_id] < self.max_samples_per_user:
                filtered_data.append(item)
                user_counts[user_id] += 1
        
        self.data = filtered_data
    
    def get_user_statistics(self) -> Dict[int, int]:
        """获取用户统计信息"""
        user_counts = {}
        for item in self.data:
            user_id = item['user_id']
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        return user_counts
    
    def get_user_samples(self, user_id: int) -> List[str]:
        """获取指定用户的所有样本路径"""
        return [item['image_path'] for item in self.data if item['user_id'] == user_id]

def create_user_data_dict(data_dir: str) -> Dict[int, List[str]]:
    """
    创建用户数据字典
    Args:
        data_dir: 数据目录路径
    Returns:
        user_data: {user_id: [image_paths]}
    """
    data_path = Path(data_dir)
    user_data = {}
    
    for user_dir in data_path.iterdir():
        if user_dir.is_dir() and user_dir.name.startswith('ID'):
            try:
                # 处理 ID1, ID_2, ID_3 等格式
                dir_name = user_dir.name
                if '_' in dir_name:
                    user_id = int(dir_name.split('_')[1])  # ID_2 -> 2
                else:
                    user_id = int(dir_name[2:])  # ID1 -> 1

                image_paths = []

                # 收集所有图像文件
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    image_paths.extend(list(user_dir.glob(ext)))

                if image_paths:
                    user_data[user_id] = [str(p) for p in image_paths]

            except ValueError:
                continue
    
    return user_data

def create_micro_doppler_dataset(
    data_dir: str,
    transform: Optional[transforms.Compose] = None,
    return_user_id: bool = False,
    user_filter: Optional[List[int]] = None,
    max_samples_per_user: Optional[int] = None,
    image_size: int = 128,
    high_quality_resize: bool = True,
) -> MicroDopplerDataset:
    """
    创建微多普勒数据集

    Args:
        data_dir: 数据目录路径
        transform: 图像变换
        return_user_id: 是否返回用户ID
        user_filter: 用户ID过滤列表
        max_samples_per_user: 每个用户的最大样本数
        image_size: 目标图像尺寸
        high_quality_resize: 是否使用高质量缩放

    Returns:
        MicroDopplerDataset实例
    """
    if transform is None:
        if high_quality_resize:
            # 高质量缩放：使用Lanczos插值 + 抗锯齿
            transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.LANCZOS,
                    antialias=True
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            # 标准缩放：双线性插值
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    return MicroDopplerDataset(
        data_dir=data_dir,
        transform=transform,
        return_user_id=return_user_id,
        user_filter=user_filter,
        max_samples_per_user=max_samples_per_user,
    )

def create_balanced_dataset(
    data_dir: str,
    samples_per_user: int = 100,
    transform: Optional[transforms.Compose] = None,
) -> MicroDopplerDataset:
    """
    创建平衡数据集
    Args:
        data_dir: 数据目录路径
        samples_per_user: 每个用户的样本数
        transform: 图像变换
    Returns:
        dataset: 平衡的数据集
    """
    return MicroDopplerDataset(
        data_dir=data_dir,
        transform=transform,
        return_user_id=True,
        max_samples_per_user=samples_per_user,
    )

def get_default_transform(
    resolution: int = 128,
    normalize: bool = True,
    interpolation: str = "lanczos"
) -> transforms.Compose:
    """
    获取默认的图像变换
    Args:
        resolution: 目标分辨率 (从256x256缩放到指定尺寸)
        normalize: 是否归一化到[-1, 1]
        interpolation: 插值方法 ("lanczos", "bicubic", "bilinear", "antialias")
    Returns:
        transform: 图像变换
    """
    # 选择插值方法
    interpolation_map = {
        "lanczos": transforms.InterpolationMode.LANCZOS,
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "antialias": transforms.InterpolationMode.BILINEAR,  # 配合antialias=True
    }

    interp_mode = interpolation_map.get(interpolation, transforms.InterpolationMode.LANCZOS)

    # 构建变换列表
    if interpolation == "antialias":
        # 使用抗锯齿缩放 (PyTorch 1.11+)
        transform_list = [
            transforms.Resize((resolution, resolution),
                            interpolation=interp_mode,
                            antialias=True),
            transforms.ToTensor(),  # [0, 1]
        ]
    else:
        transform_list = [
            transforms.Resize((resolution, resolution), interpolation=interp_mode),
            transforms.ToTensor(),  # [0, 1]
        ]

    if normalize:
        # 归一化到[-1, 1]，适配VQ-VAE训练
        transform_list.append(
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )

    return transforms.Compose(transform_list)

def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    反归一化tensor从[-1, 1]到[0, 1]
    Args:
        tensor: 归一化的tensor [-1, 1]
    Returns:
        tensor: 反归一化的tensor [0, 1]
    """
    return (tensor + 1.0) / 2.0

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    将tensor转换为PIL图像
    Args:
        tensor: [C, H, W] tensor，范围[-1, 1]或[0, 1]
    Returns:
        image: PIL图像
    """
    # 如果是[-1, 1]范围，先反归一化
    if tensor.min() < 0:
        tensor = denormalize_tensor(tensor)

    # 转换为PIL
    tensor = torch.clamp(tensor, 0, 1)
    tensor = (tensor * 255).byte()

    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

    return Image.fromarray(tensor.cpu().numpy())

class VQTokenDataset(Dataset):
    """
    VQ Token数据集
    用于Transformer训练，包含预计算的VQ tokens
    """
    
    def __init__(
        self,
        token_data: List[Dict],
        max_seq_len: int = 256,
        pad_token_id: int = 1024,
    ):
        """
        Args:
            token_data: token数据列表，每个元素包含user_id和tokens
            max_seq_len: 最大序列长度
            pad_token_id: 填充token ID
        """
        self.token_data = token_data
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        print(f"📊 VQ Token数据集:")
        print(f"   总序列数: {len(self.token_data)}")
        print(f"   最大序列长度: {self.max_seq_len}")
    
    def __len__(self):
        return len(self.token_data)
    
    def __getitem__(self, idx):
        item = self.token_data[idx]
        
        user_id = torch.tensor(item['user_id'], dtype=torch.long)
        tokens = item['tokens']
        
        # 截断或填充到固定长度
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            pad_length = self.max_seq_len - len(tokens)
            pad_tokens = torch.full((pad_length,), self.pad_token_id, dtype=tokens.dtype)
            tokens = torch.cat([tokens, pad_tokens])
        
        return {
            'user_id': user_id,
            'tokens': tokens,
        }

def create_vq_token_dataset(
    vqvae_model,
    data_dir: str,
    transform: transforms.Compose,
    max_seq_len: int = 256,
    device: str = "cuda",
) -> VQTokenDataset:
    """
    创建VQ Token数据集
    Args:
        vqvae_model: 预训练的VQ-VAE模型
        data_dir: 数据目录
        transform: 图像变换
        max_seq_len: 最大序列长度
        device: 计算设备
    Returns:
        dataset: VQ Token数据集
    """
    from tqdm import tqdm
    
    # 创建基础数据集
    base_dataset = MicroDopplerDataset(
        data_dir=data_dir,
        transform=transform,
        return_user_id=True,
    )
    
    # 预计算tokens
    vqvae_model.eval()
    token_data = []
    
    with torch.no_grad():
        for i in tqdm(range(len(base_dataset)), desc="预计算VQ tokens"):
            image, user_id = base_dataset[i]
            image_tensor = image.unsqueeze(0).to(device)
            
            # 编码为tokens
            result = vqvae_model.encode(image_tensor, return_dict=True)
            tokens = result['encoding_indices'].flatten().cpu()
            
            token_data.append({
                'user_id': user_id,
                'tokens': tokens,
            })
    
    return VQTokenDataset(token_data, max_seq_len)

def create_stratified_split(dataset, train_ratio=0.8, val_ratio=0.2, random_seed=42):
    """
    创建分层数据集划分，确保每个用户都在训练集和验证集中

    Args:
        dataset: 数据集对象
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_seed: 随机种子

    Returns:
        train_dataset, val_dataset
    """
    print("🔄 执行分层数据集划分...")

    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 按用户分组数据
    user_indices = defaultdict(list)

    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]

            # 处理不同的数据格式
            if isinstance(sample, dict):
                user_id = sample['user_id']
            elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                if isinstance(sample[1], torch.Tensor):
                    user_id = sample[1].item()
                else:
                    user_id = sample[1]
            else:
                print(f"⚠️ 跳过未知格式的样本: {type(sample)}")
                continue

            user_indices[user_id].append(idx)

        except Exception as e:
            print(f"⚠️ 处理样本{idx}时出错: {e}")
            continue

    print(f"📊 发现 {len(user_indices)} 个用户")
    for user_id, indices in user_indices.items():
        print(f"   用户{user_id}: {len(indices)}个样本")

    # 为每个用户分配训练集和验证集样本
    train_indices = []
    val_indices = []

    for user_id, indices in user_indices.items():
        # 随机打乱该用户的样本
        random.shuffle(indices)

        # 计算分割点
        n_samples = len(indices)
        n_train = max(1, int(n_samples * train_ratio))  # 至少1个训练样本
        n_val = max(1, n_samples - n_train)  # 至少1个验证样本

        # 如果样本太少，调整分配
        if n_samples < 2:
            train_indices.extend(indices)
            val_indices.extend(indices)  # 复制到验证集
            print(f"   ⚠️ 用户{user_id}样本太少({n_samples})，训练集和验证集共享")
        else:
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            print(f"   ✅ 用户{user_id}: 训练集{n_train}个, 验证集{n_val}个")

    # 创建子数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"\n📊 数据集划分完成:")
    print(f"   总样本数: {len(dataset)}")
    print(f"   训练集: {len(train_dataset)} ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"   验证集: {len(val_dataset)} ({len(val_dataset)/len(dataset)*100:.1f}%)")

    return train_dataset, val_dataset

def create_datasets_with_split(data_dir, train_ratio=0.8, val_ratio=0.2, return_user_id=True, random_seed=42, image_size=128, high_quality_resize=True):
    """
    创建带有自动划分的数据集

    Args:
        data_dir: 数据目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        return_user_id: 是否返回用户ID
        random_seed: 随机种子
        image_size: 目标图像尺寸
        high_quality_resize: 是否使用高质量缩放

    Returns:
        train_dataset, val_dataset
    """
    print("🚀 创建带有自动划分的数据集...")

    # 创建完整数据集
    full_dataset = create_micro_doppler_dataset(
        data_dir=data_dir,
        return_user_id=return_user_id,
        image_size=image_size,
        high_quality_resize=high_quality_resize
    )

    # 执行分层划分
    train_dataset, val_dataset = create_stratified_split(
        dataset=full_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )

    return train_dataset, val_dataset
