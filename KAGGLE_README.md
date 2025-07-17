# Kaggle环境 - 微多普勒时频图数据增广

专为您的Kaggle数据集优化的训练流程。

## 📋 数据集信息

- **路径**: `/kaggle/input/dataset`
- **结构**: `ID_1/`, `ID_2/`, ..., `ID_31/`
- **用户数**: 31位用户
- **图像**: 256×256 彩色微多普勒时频图

## 🚀 快速开始

### 0. 环境测试 (强烈推荐)

在开始训练前，请先运行测试确保环境兼容：

```bash
# 1. 克隆项目
!git clone https://github.com/heimaoqqq/VAE.git
%cd VAE

# 2. 运行完整环境测试
!python test_kaggle_environment.py

# 3. 或分步测试
!python test_dependencies.py  # 依赖版本测试
!python test_diffusers_compatibility.py  # API兼容性测试
!python kaggle_config.py  # 数据集验证
```

### 1. 一键训练 (推荐)

```bash
# 运行完整训练流程
python train_kaggle.py --stage all

# 或分阶段运行
python train_kaggle.py --stage setup    # 环境设置
python train_kaggle.py --stage vae      # VAE训练
python train_kaggle.py --stage diffusion # 扩散训练
python train_kaggle.py --stage generate  # 生成图像
```

### 2. 手动训练

```bash
# 1. 环境设置和验证
python kaggle_config.py

# 2. VAE训练 (约2-3小时)
python training/train_vae.py \
    --data_dir /kaggle/input/dataset \
    --output_dir /kaggle/working/outputs/vae \
    --batch_size 8 \
    --num_epochs 50 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 2

# 3. 扩散训练 (约4-6小时)
python training/train_diffusion.py \
    --data_dir /kaggle/input/dataset \
    --vae_path /kaggle/working/outputs/vae/final_model \
    --output_dir /kaggle/working/outputs/diffusion \
    --batch_size 4 \
    --num_epochs 100 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 4

# 4. 生成图像
python inference/generate.py \
    --vae_path /kaggle/working/outputs/vae/final_model \
    --unet_path /kaggle/working/outputs/diffusion/final_model/unet \
    --condition_encoder_path /kaggle/working/outputs/diffusion/final_model/condition_encoder.pt \
    --num_users 31 \
    --user_ids 1 5 10 15 20 25 31 \
    --num_images_per_user 5
```

## ⚙️ Kaggle优化配置

### 内存优化
- **批次大小**: VAE=8, 扩散=4
- **混合精度**: FP16
- **梯度累积**: 减少内存使用
- **工作进程**: 2个 (Kaggle限制)

### 时间优化
- **VAE轮数**: 50 (原100)
- **扩散轮数**: 100 (原200)
- **保存间隔**: 更频繁的检查点
- **推理步数**: 50步快速生成

### 存储优化
- **输出路径**: `/kaggle/working/outputs`
- **检查点**: 定期保存，防止丢失
- **样本图像**: 训练过程中生成

## 📊 预期性能

| 阶段 | 时间 | GPU内存 | 输出 |
|------|------|---------|------|
| VAE训练 | 2-3小时 | ~6GB | 重建模型 |
| 扩散训练 | 4-6小时 | ~8GB | 生成模型 |
| 图像生成 | 5-10分钟 | ~4GB | 样本图像 |
| **总计** | **6-9小时** | **8GB** | **完整模型** |

## 🔍 监控训练

### VAE训练指标
```python
# 目标值
loss/recon < 0.01      # 重建损失
loss/kl < 0.001        # KL散度
loss/perceptual < 0.1  # 感知损失
loss/freq < 0.05       # 频域损失
```

### 扩散训练指标
```python
# 目标值
loss < 0.1             # 扩散损失 (逐渐下降)
val_loss ≈ loss        # 验证损失接近训练损失
```

## 📁 输出结构

```
/kaggle/working/outputs/
├── vae/
│   ├── final_model/           # VAE模型
│   ├── checkpoints/           # 训练检查点
│   └── samples/               # 重建样本
├── diffusion/
│   ├── final_model/
│   │   ├── unet/             # UNet模型
│   │   └── condition_encoder.pt # 条件编码器
│   ├── checkpoints/
│   └── samples/               # 生成样本
└── generated_images/
    ├── ID_1/                  # 用户1生成图像
    ├── ID_5/
    └── ...
```

## 🛠️ 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 解决方案
   --batch_size 4
   --gradient_accumulation_steps 4
   ```

2. **训练时间过长**
   ```bash
   # 解决方案
   --num_epochs 30  # 减少轮数
   --save_interval 5  # 更频繁保存
   ```

3. **数据加载错误**
   ```bash
   # 检查数据结构
   python kaggle_config.py
   ```

4. **模型不收敛**
   ```bash
   # 调整学习率
   --learning_rate 5e-5
   ```

### 检查点恢复

如果训练中断，可以从检查点恢复：

```bash
# VAE恢复
python training/train_vae.py \
    --resume_from_checkpoint /kaggle/working/outputs/vae/checkpoints/checkpoint_epoch_20.pt \
    # ... 其他参数

# 扩散恢复
python training/train_diffusion.py \
    --resume_from_checkpoint /kaggle/working/outputs/diffusion/checkpoints/checkpoint_epoch_50.pt \
    # ... 其他参数
```

## 📈 结果验证

### 1. 检查VAE重建质量
```bash
# 查看重建样本
ls /kaggle/working/outputs/vae/samples/
```

### 2. 检查扩散生成质量
```bash
# 查看生成样本
ls /kaggle/working/outputs/diffusion/samples/
```

### 3. 评估最终结果
```bash
# 运行评估
python utils/metrics.py \
    --real_dir /kaggle/input/dataset \
    --generated_dir /kaggle/working/outputs/generated_images
```

## 💡 优化建议

### 提升质量
1. **增加训练轮数** (如果时间允许)
2. **调整损失权重** (KL权重、感知损失权重)
3. **使用更多推理步数** (生成时)

### 节省时间
1. **使用预训练VAE** (如果有)
2. **减少验证频率**
3. **使用DDIM调度器** (更快推理)

### 节省内存
1. **启用梯度检查点**
2. **使用更小的交叉注意力维度**
3. **减少UNet层数**

## 🎯 成功标准

训练成功的标志：
- ✅ VAE能够清晰重建输入图像
- ✅ 扩散损失稳定下降
- ✅ 生成图像具有明显的用户特征差异
- ✅ 条件控制准确 (指定用户ID生成对应特征)

## 📞 技术支持

如果遇到问题：
1. 检查 `kaggle_config.py` 输出的数据集统计
2. 监控GPU内存使用情况
3. 查看训练日志中的损失曲线
4. 验证生成的样本图像质量

祝您在Kaggle上训练顺利！🚀
