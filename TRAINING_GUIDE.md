# 微多普勒时频图数据增广项目 - 完整训练指南

## 🚀 快速开始

### 1. 环境设置

```bash
# 运行环境设置脚本
python setup_environment.py

# 或手动安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将您的微多普勒数据按以下结构组织：

```
data/
├── user_01/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── user_02/
│   ├── image_001.png
│   └── ...
└── user_31/
    └── ...
```

**要求**:
- 图像格式：PNG/JPG
- 图像尺寸：256×256 (会自动调整)
- 用户目录命名：`user_XX` (XX为两位数字)

**⚠️ 关于数据增强**:
- 微多普勒时频图对传统数据增强(旋转、翻转、颜色调整)很敏感
- 本项目的**数据增广**是通过生成式AI创建新样本，不是传统增强
- 建议训练时不使用 `--use_augmentation` 参数
- 详见：[微多普勒数据增广说明](MICRO_DOPPLER_AUGMENTATION.md)

## 📋 训练流程

### 阶段1: VAE训练 (2-3天)

```bash
# 基础训练
python training/train_vae.py \
    --data_dir ./data \
    --output_dir ./outputs/vae \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    # --use_augmentation \  # 微多普勒图像建议不使用传统增强
    --use_wandb

# 高级配置
python training/train_vae.py \
    --data_dir ./data \
    --output_dir ./outputs/vae \
    --batch_size 8 \
    --num_epochs 150 \
    --learning_rate 1e-4 \
    --kl_weight 1e-6 \
    --perceptual_weight 0.1 \
    --freq_weight 0.05 \
    --mixed_precision fp16 \
    # --use_augmentation \  # 微多普勒图像建议不使用传统增强
    --use_wandb \
    --experiment_name "vae_optimized"
```

**监控指标**:
- `loss/recon`: 重建损失 (目标: < 0.01)
- `loss/kl`: KL散度损失 (目标: 稳定在小值)
- `loss/perceptual`: 感知损失 (目标: < 0.1)
- `loss/freq`: 频域损失 (目标: < 0.05)

### 阶段2: 条件扩散训练 (3-5天)

```bash
# 基础训练
python training/train_diffusion.py \
    --data_dir ./data \
    --vae_path ./outputs/vae/final_model \
    --output_dir ./outputs/diffusion \
    --batch_size 8 \
    --num_epochs 200 \
    --learning_rate 1e-4 \
    --use_wandb

# 高级配置
python training/train_diffusion.py \
    --data_dir ./data \
    --vae_path ./outputs/vae/final_model \
    --output_dir ./outputs/diffusion \
    --batch_size 4 \
    --num_epochs 300 \
    --learning_rate 1e-4 \
    --cross_attention_dim 768 \
    --num_train_timesteps 1000 \
    --condition_dropout 0.1 \
    --gradient_accumulation_steps 2 \
    --mixed_precision fp16 \
    --use_wandb \
    --experiment_name "diffusion_optimized"
```

**监控指标**:
- `loss`: 扩散损失 (目标: 逐渐下降)
- `val_loss`: 验证损失 (目标: 与训练损失接近)

## 🎯 生成图像

### 基础生成

```bash
# 为指定用户生成图像
python inference/generate.py \
    --vae_path ./outputs/vae/final_model \
    --unet_path ./outputs/diffusion/final_model/unet \
    --condition_encoder_path ./outputs/diffusion/final_model/condition_encoder.pt \
    --num_users 31 \
    --user_ids 1 5 10 15 \
    --num_images_per_user 5 \
    --output_dir ./generated_images
```

### 高质量生成

```bash
# 使用更多推理步数和引导
python inference/generate.py \
    --vae_path ./outputs/vae/final_model \
    --unet_path ./outputs/diffusion/final_model/unet \
    --condition_encoder_path ./outputs/diffusion/final_model/condition_encoder.pt \
    --num_users 31 \
    --user_ids 1 2 3 4 5 \
    --num_images_per_user 10 \
    --num_inference_steps 100 \
    --guidance_scale 7.5 \
    --scheduler_type ddim \
    --output_dir ./generated_images_hq
```

### 用户间插值

```bash
# 生成用户间的插值图像
python inference/generate.py \
    --vae_path ./outputs/vae/final_model \
    --unet_path ./outputs/diffusion/final_model/unet \
    --condition_encoder_path ./outputs/diffusion/final_model/condition_encoder.pt \
    --num_users 31 \
    --interpolation \
    --interpolation_users 1 10 \
    --interpolation_steps 15 \
    --output_dir ./interpolation_results
```

## 📊 模型评估

```bash
# 评估生成质量
python utils/metrics.py \
    --real_dir ./data \
    --generated_dir ./generated_images \
    --device cuda
```

**评估指标**:
- **FID**: 越低越好 (目标: < 50)
- **LPIPS**: 感知相似性
- **频域相似性**: 针对时频图的特殊指标

## ⚙️ 超参数调优建议

### VAE训练优化

```python
# 如果重建质量不好
--kl_weight 1e-8  # 降低KL权重
--perceptual_weight 0.2  # 增加感知损失权重

# 如果训练不稳定
--learning_rate 5e-5  # 降低学习率
--batch_size 8  # 减小批次大小

# 如果内存不足
--mixed_precision fp16
--gradient_accumulation_steps 2
--batch_size 4
```

### 扩散训练优化

```python
# 如果生成质量不好
--num_train_timesteps 1000  # 增加时间步数
--guidance_scale 7.5  # 调整引导强度

# 如果条件控制不准确
--condition_dropout 0.05  # 降低条件dropout
--cross_attention_dim 1024  # 增加条件维度

# 如果训练太慢
--num_inference_steps 50  # 推理时使用较少步数
--scheduler_type ddim  # 使用DDIM调度器
```

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 解决方案
   --batch_size 4
   --gradient_accumulation_steps 4
   --mixed_precision fp16
   ```

2. **VAE重建模糊**
   ```bash
   # 解决方案
   --kl_weight 1e-8
   --perceptual_weight 0.2
   --freq_weight 0.1
   ```

3. **扩散训练不收敛**
   ```bash
   # 解决方案
   --learning_rate 5e-5
   --num_train_timesteps 1000
   --condition_dropout 0.05
   ```

4. **生成图像质量差**
   ```bash
   # 解决方案
   --num_inference_steps 100
   --guidance_scale 7.5
   --scheduler_type ddim
   ```

### 检查点恢复

```bash
# 从检查点恢复VAE训练
python training/train_vae.py \
    --resume_from_checkpoint ./outputs/vae/checkpoints/checkpoint_epoch_50.pt \
    # ... 其他参数

# 从检查点恢复扩散训练
python training/train_diffusion.py \
    --resume_from_checkpoint ./outputs/diffusion/checkpoints/checkpoint_epoch_100.pt \
    # ... 其他参数
```

## 📈 性能基准

### 硬件要求

| 配置 | GPU内存 | 训练时间 | 批次大小 |
|------|---------|----------|----------|
| 最低配置 | 8GB | VAE: 3天, 扩散: 5天 | 4 |
| 推荐配置 | 16GB | VAE: 2天, 扩散: 3天 | 8 |
| 高端配置 | 24GB+ | VAE: 1天, 扩散: 2天 | 16+ |

### 预期结果

| 指标 | 目标值 | 说明 |
|------|--------|------|
| VAE重建PSNR | > 25 dB | 重建质量 |
| VAE重建SSIM | > 0.8 | 结构相似性 |
| FID | < 50 | 生成质量 |
| 训练收敛 | < 100 epochs | 扩散训练 |

## 🎉 完成检查清单

- [ ] 环境设置完成
- [ ] 数据格式正确
- [ ] VAE训练完成且重建质量满意
- [ ] 扩散训练完成且损失收敛
- [ ] 生成图像质量评估通过
- [ ] 条件控制准确性验证
- [ ] 模型保存和部署准备

## 📞 技术支持

如果遇到问题，请检查：
1. 数据格式是否正确
2. GPU内存是否充足
3. 依赖包是否正确安装
4. 超参数是否合理

祝您训练顺利！🚀
