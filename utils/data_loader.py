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
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

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
                return image, item['user_id']
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

def get_default_transform(resolution: int = 128, normalize: bool = True) -> transforms.Compose:
    """
    获取默认的图像变换
    Args:
        resolution: 目标分辨率 (从256x256缩放到指定尺寸)
        normalize: 是否归一化到[-1, 1]
    Returns:
        transform: 图像变换
    """
    transform_list = [
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
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
