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

## 🔍 验证系统

### 核心验证工具
- **用户分类器**: 基于ResNet-18的二分类器，验证生成图像是否包含用户特征
- **极端指导强度**: 针对微多普勒数据的特殊生成策略
- **成功标准**: 置信度>0.8算成功，成功率≥60%为良好

### 微多普勒数据特点
- **用户间差异小**: 步态特征的微小差异需要精细学习
- **需要极端参数**: 指导强度30-50，推理步数150-200
- **频域特征**: 重点关注频率重心、扩散、峰值位置差异

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

# 6. 如果质量不佳，使用诊断工具
!python diagnose_vae.py     # 分析问题原因
!python quick_test_vae.py   # 测试新配置

# 7. 条件扩散训练
!python training/train_diffusion.py \
    --data_dir "/kaggle/input/dataset" \
    --vae_path "/kaggle/input/final-model" \
    --output_dir "/kaggle/working/diffusion_model" \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-4

# 8. 生成图像 (修复后的推理脚本 - 支持分类器自由指导)
!python inference/generate_training_style.py \
    --vae_path "/kaggle/input/final-model" \
    --unet_path "/kaggle/working/diffusion_model" \
    --condition_encoder_path "/kaggle/working/diffusion_model/condition_encoder.pt" \
    --num_users 31 \
    --user_ids 1 5 10 15 \
    --num_images_per_user 16 \
    --num_inference_steps 50 \
    --guidance_scale 15.0 \
    --data_dir "/kaggle/input/dataset" \
    --output_dir "/kaggle/working/generated_images"

# 9. 现代化验证系统 (推荐) - 自动检查用户ID映射一致性
!python validation/validation_pipeline.py \
    --target_user_id 1 \
    --real_data_root "/kaggle/input/dataset" \
    --generate_images \
    --vae_path "/kaggle/input/final-model" \
    --unet_path "/kaggle/working/diffusion_model" \
    --condition_encoder_path "/kaggle/working/diffusion_model/condition_encoder.pt" \
    --model_type microdoppler \
    --classifier_epochs 30 \
    --max_samples_per_class 1000 \
    --guidance_scale 15.0 \
    --num_inference_steps 50

# 9. 极端指导强度测试 (微多普勒数据特化)
!python validation/validation_pipeline.py \
    --target_user_id 1 \
    --real_data_root "/kaggle/input/dataset" \
    --generate_images \
    --vae_path "/kaggle/input/final-model" \
    --unet_path "/kaggle/working/diffusion_model" \
    --condition_encoder_path "/kaggle/working/diffusion_model/condition_encoder.pt" \
    --model_type microdoppler \
    --guidance_scale 35.0 \
    --num_inference_steps 150 \
    --classifier_epochs 40

# 10. 对比测试 (ResNet vs 微多普勒专用CNN)
!python validation/validation_pipeline.py \
    --target_user_id 1 \
    --real_data_root "/kaggle/input/dataset" \
    --model_type resnet \
    --classifier_epochs 30

# 7. 扩散模型训练 (可选)
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
├── diagnose_vae.py       # VAE问题诊断工具
├── quick_test_vae.py     # 快速配置测试
├── install_lpips.py      # 感知损失管理
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

### 基础问题
1. **依赖冲突**: 运行 `python ultimate_fix_kaggle.py`
2. **感知损失问题**: 运行 `python install_lpips.py` (自动处理)
3. **CUDA内存不足**: 减小batch_size
4. **训练中断**: 检查数据路径和格式

### VAE重建质量差 (PSNR < 20dB)
```bash
# 1. 诊断问题
python diagnose_vae.py      # 分析损失组成和模型行为

# 2. 测试新配置
python quick_test_vae.py    # 验证参数设置

# 3. 常见修复
# - KL权重过高: 1e-4 → 1e-6
# - 学习率过高: 2e-4 → 1e-4
# - 感知损失设备问题: 权重1.0 → 0.1
# - 训练不足: 增加轮数到80+
```

## 📊 验证系统

### 验证流程
1. **统计验证**: 检查生成图像的基本统计特性
2. **度量学习验证**: 使用Siamese网络学习用户相似性
3. **分类器验证**: 训练专用分类器判断生成质量
4. **对比控制实验**: 验证条件生成的有效性

### 关键改进
- **分层负样本采样**: 确保每个其他用户都有代表性样本
- **负样本比例优化**: 从3:1提升到8:1，充分覆盖用户多样性
- **对比控制实验**: 生成错误条件图像进行对比验证

### 验证指标
- **成功率**: 生成图像被正确识别的比例 (目标 > 60%)
- **置信度**: 分类器的平均置信度 (目标 > 0.8)
- **条件控制比**: 正确条件 vs 错误条件的成功率比值 (目标 > 2.0)

## 🎯 使用建议

- **新手**: 直接使用 `train_celeba_standard.py`
- **进阶**: 使用 `training/` 目录下的专业脚本
- **质量检查**: 训练后运行 `check_vae.py`
- **验证生成**: 使用 `validation/validation_pipeline.py` 全面验证

## 📄 许可证

MIT License
