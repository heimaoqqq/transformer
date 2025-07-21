# 🚀 VQ-VAE + Transformer 训练指南

## 📋 训练流程概述

本项目采用**两阶段训练**策略：

1. **阶段1**: 训练VQ-VAE学习图像的离散表示
2. **阶段2**: 训练Transformer从用户ID生成token序列

## 🎯 快速开始

### 方法1: 完整自动训练 (推荐)

```bash
# 在Kaggle中运行
python train_main.py --data_dir /kaggle/input/dataset
```

### 方法2: 分步训练

```bash
# 步骤1: 仅训练VQ-VAE
python train_main.py --skip_transformer --data_dir /kaggle/input/dataset

# 步骤2: 仅训练Transformer (需要先完成步骤1)
python train_main.py --skip_vqvae --data_dir /kaggle/input/dataset
```

## 🔧 详细训练参数

### 基础参数
```bash
python train_main.py \
    --data_dir /kaggle/input/dataset \
    --output_dir /kaggle/working/outputs \
    --resolution 128 \
    --num_users 31 \
    --codebook_size 1024
```

### VQ-VAE参数
```bash
python train_main.py \
    --vqvae_epochs 50 \
    --vqvae_lr 1e-4 \
    --commitment_cost 0.25 \
    --ema_decay 0.99 \
    --interpolation lanczos
```

### Transformer参数
```bash
python train_main.py \
    --transformer_epochs 100 \
    --transformer_lr 5e-5 \
    --n_embd 512 \
    --n_layer 8 \
    --n_head 8 \
    --use_cross_attention
```

## 📁 输出文件结构

训练完成后会生成以下文件结构：

```
/kaggle/working/outputs/vqvae_transformer/
├── vqvae/                          # VQ-VAE模型
│   ├── final_model/                # diffusers格式 (推荐)
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   ├── best_model.pth              # 最佳checkpoint
│   ├── checkpoint_epoch_XXX.pth    # 训练checkpoints
│   └── samples/                    # 重建样本
└── transformer/                    # Transformer模型
    ├── final_model/                # 最终模型
    ├── best_model.pth              # 最佳checkpoint
    ├── checkpoint_epoch_XXX.pth    # 训练checkpoints
    └── generated_samples/          # 生成样本
```

## ✅ 模型兼容性检查

### VQ-VAE训练完成后检查
```python
# 检查VQ-VAE输出文件
import os
from pathlib import Path

vqvae_path = Path("/kaggle/working/outputs/vqvae_transformer/vqvae")

print("📁 VQ-VAE输出文件:")
if (vqvae_path / "final_model").exists():
    print("✅ final_model/ (diffusers格式)")
if (vqvae_path / "best_model.pth").exists():
    print("✅ best_model.pth (checkpoint格式)")

# 测试VQ-VAE加载
from models.vqvae_model import MicroDopplerVQVAE
try:
    model = MicroDopplerVQVAE.from_pretrained(vqvae_path / "final_model")
    print("✅ VQ-VAE模型加载成功")
except:
    print("⚠️ diffusers格式加载失败，将使用checkpoint格式")
```

### Transformer训练前检查
```python
# 验证VQ-VAE模型可用性
python -c "
from training.train_transformer import TransformerTrainer
import argparse

# 模拟参数
args = argparse.Namespace(
    vqvae_path='/kaggle/working/outputs/vqvae_transformer/vqvae',
    data_dir='/kaggle/input/dataset',
    output_dir='/tmp/test',
    codebook_size=1024,
    num_users=31
)

try:
    trainer = TransformerTrainer(args)
    print('✅ Transformer训练器初始化成功')
except Exception as e:
    print(f'❌ 初始化失败: {e}')
"
```

## 🎮 GPU优化配置

项目会自动检测GPU并优化配置：

### Tesla P100 (16GB)
- VQ-VAE batch_size: 16
- Transformer batch_size: 8
- 混合精度: 关闭

### Tesla T4 (16GB)
- VQ-VAE batch_size: 12
- Transformer batch_size: 6
- 混合精度: 开启

### 其他GPU (≥8GB)
- VQ-VAE batch_size: 8
- Transformer batch_size: 4
- 混合精度: 开启

## 🔍 训练监控

### VQ-VAE训练监控
- **重建质量**: PSNR, SSIM指标
- **码本使用**: 使用率统计
- **损失曲线**: VQ损失 + 重建损失

### Transformer训练监控
- **生成质量**: 定期生成样本
- **损失曲线**: 交叉熵损失
- **用户条件**: 验证用户特征保持

## ⚠️ 常见问题

### 1. VQ-VAE训练失败
```bash
# 检查数据集格式
python test_dataset.py

# 降低batch_size
python train_main.py --vqvae_epochs 10 --batch_size 4
```

### 2. Transformer找不到VQ-VAE模型
```bash
# 检查VQ-VAE输出
ls -la /kaggle/working/outputs/vqvae_transformer/vqvae/

# 手动指定VQ-VAE路径
python training/train_transformer.py \
    --vqvae_path /kaggle/working/outputs/vqvae_transformer/vqvae \
    --data_dir /kaggle/input/dataset
```

### 3. 内存不足
```bash
# 减小batch_size和模型尺寸
python train_main.py \
    --resolution 64 \
    --codebook_size 512 \
    --n_embd 256 \
    --n_layer 4
```

## 🎯 推荐训练策略

### 快速验证 (10分钟)
```bash
python train_main.py \
    --vqvae_epochs 5 \
    --transformer_epochs 10 \
    --data_dir /kaggle/input/dataset
```

### 标准训练 (2-3小时)
```bash
python train_main.py \
    --vqvae_epochs 50 \
    --transformer_epochs 100 \
    --data_dir /kaggle/input/dataset
```

### 高质量训练 (6-8小时)
```bash
python train_main.py \
    --vqvae_epochs 100 \
    --transformer_epochs 200 \
    --vqvae_lr 5e-5 \
    --transformer_lr 1e-5 \
    --data_dir /kaggle/input/dataset
```

## 📊 训练完成后

训练完成后，可以使用以下脚本进行生成和验证：

```bash
# 生成新样本
python generate_main.py \
    --model_dir /kaggle/working/outputs/vqvae_transformer \
    --target_user_id 1 \
    --num_samples 10

# 验证模型质量
python validate_main.py \
    --model_dir /kaggle/working/outputs/vqvae_transformer \
    --real_data_dir /kaggle/input/dataset
```

🎉 **现在您可以开始训练了！**
