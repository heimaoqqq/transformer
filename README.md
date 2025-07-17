# 微多普勒时频图数据增广项目

基于 Diffusers 的两阶段扩散模型，用于微多普勒时频图的数据增广。

## 🎯 项目概述

- **数据集**: 256×256 彩色微多普勒时频图像
- **用户数**: 31位用户的步态数据
- **目标**: 通过条件扩散生成指定用户的步态微多普勒时频图像
- **技术栈**: PyTorch + Diffusers + VAE + UNet

## 🏗️ 技术方案

### 第一阶段: VAE (变分自编码器)
- 将 256×256 图像编码到潜在空间 (32×32×4)
- 学习连续的视觉表示
- 压缩数据维度，提高训练效率
- 使用 AutoencoderKL (Stable Diffusion 同款)

### 第二阶段: 条件扩散
- 在潜在空间中进行扩散训练
- 以用户ID作为条件信息 (交叉注意力)
- 生成指定用户的微多普勒图像
- 使用 UNet2DConditionModel

## 🚀 快速开始

### Kaggle环境 (推荐)

```bash
# 1. 克隆项目
!git clone https://github.com/heimaoqqq/VAE.git
%cd VAE

# 2. 修复依赖 (如果需要)
!python ultimate_fix_kaggle.py

# 3. 设置感知损失 (自动安装LPIPS并启用感知损失)
!python install_lpips.py

# 4. VAE训练 (64×64分辨率，高质量配置)
!python train_celeba_standard.py

# 5. 检查训练质量
!python check_vae.py

# 6. 扩散模型训练 (可选)
!python training/train_diffusion.py \
    --data_dir /kaggle/input/dataset \
    --vae_path /kaggle/working/outputs/vae_celeba_standard/final_model \
    --output_dir /kaggle/working/outputs/diffusion
```

### 本地环境

#### 1. 环境设置
```bash
pip install -r requirements.txt
```

#### 2. 准备数据
支持两种数据结构：

**Kaggle格式**:
```
/kaggle/input/dataset/
├── ID_1/
├── ID_2/
└── ID_31/
```

**标准格式**:
```
data/
├── user_01/
├── user_02/
└── user_31/
```

#### 3. 训练模型

```bash
# 第一阶段: VAE训练
python training/train_vae.py --data_dir ./data --output_dir ./outputs/vae

# 第二阶段: 条件扩散训练
python training/train_diffusion.py \
    --data_dir ./data \
    --vae_path ./outputs/vae/final_model \
    --output_dir ./outputs/diffusion
```

#### 4. 生成图像
```bash
# 生成指定用户的图像
python inference/generate.py \
    --vae_path ./outputs/vae/final_model \
    --unet_path ./outputs/diffusion/final_model/unet \
    --condition_encoder_path ./outputs/diffusion/final_model/condition_encoder.pt \
    --num_users 31 \
    --user_ids 1 5 10 \
    --num_images_per_user 5
```

## 📁 项目结构

```
├── training/              # 训练脚本
│   ├── train_vae.py      # VAE训练核心
│   └── train_diffusion.py # 条件扩散训练
├── inference/             # 推理脚本
│   └── generate.py       # 条件生成
├── utils/                 # 工具函数
│   ├── data_loader.py    # 数据加载器
│   └── metrics.py        # 评估指标
├── train_celeba_standard.py # 主训练脚本 (推荐)
├── check_vae.py          # VAE质量检查
├── ultimate_fix_kaggle.py # 依赖修复工具
└── requirements.txt      # 依赖管理
```

## 📋 环境要求

- **Python**: 3.8+
- **PyTorch**: 2.1.0
- **GPU**: 8GB+ VRAM (Kaggle T4)
- **依赖**: 见 requirements.txt

## 📊 性能基准 (Kaggle环境)

| 阶段 | 时间 | GPU内存 | 输出质量 |
|------|------|---------|----------|
| VAE训练 (改进版) | 2-3小时 | ~6GB | PSNR > 25dB |
| 扩散训练 | 4-6小时 | ~8GB | 条件生成 |
| 图像生成 | 5-10分钟 | ~4GB | 用户特征 |

### 🔧 改进内容
- **KL权重**: 1e-6 → 1e-4 (Stable Diffusion标准)
- **感知损失**: 禁用 → 启用 (权重1.0)
- **频域损失**: 0.05 → 0.1 (增强时频特征保持)
- **训练轮数**: 30 → 50 (提升收敛质量)
- **学习率调度**: 添加warmup策略

## 🔧 故障排除

1. **依赖冲突**: 运行 `python ultimate_fix_kaggle.py`
2. **感知损失问题**: 运行 `python install_lpips.py` (自动处理)
3. **CUDA内存不足**: 减小batch_size
4. **VAE重建质量差**:
   - 确保感知损失已启用 (perceptual_weight=1.0)
   - 检查KL权重设置 (推荐1e-4)
   - 增加训练轮数
5. **训练中断**: 检查数据路径和格式

## 🎯 使用建议

- **新手**: 直接使用 `train_celeba_standard.py`
- **进阶**: 使用 `training/` 目录下的专业脚本
- **质量检查**: 训练后运行 `check_vae.py`

## 📄 许可证

MIT License
