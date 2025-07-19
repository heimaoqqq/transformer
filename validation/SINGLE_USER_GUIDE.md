# 单用户验证指南

## 🎯 适用场景

当你的显存有限或者只想验证特定用户时，可以使用单用户验证方案：
- 只训练一个用户的分类器
- 只生成一个用户的图像
- 大大节省显存和时间

## 🚀 快速开始

### 方案1: 完整工作流程 (训练+生成+验证)

```bash
python validation/single_user_validation.py \
    --target_user_id 1 \
    --real_data_root "/kaggle/input/dataset" \
    --output_dir "/kaggle/working/user_1_validation" \
    --generate_images \
    --vae_path "/kaggle/input/final-model" \
    --unet_path "/kaggle/input/diffusion-final-model" \
    --condition_encoder_path "/kaggle/input/diffusion-final-model/condition_encoder.pt" \
    --epochs 15 \
    --batch_size 16 \
    --num_images_to_generate 8
```

### 方案2: 只训练分类器

```bash
python validation/single_user_validation.py \
    --target_user_id 1 \
    --real_data_root "/kaggle/input/dataset" \
    --output_dir "/kaggle/working/user_1_validation" \
    --epochs 15 \
    --batch_size 16
```

### 方案3: 验证已有生成图像

```bash
python validation/single_user_validation.py \
    --target_user_id 1 \
    --real_data_root "/kaggle/input/dataset" \
    --generated_images_dir "/kaggle/working/generated_images/user_01" \
    --output_dir "/kaggle/working/user_1_validation" \
    --epochs 15 \
    --batch_size 16
```

## 📊 显存优化配置

针对显存有限的环境，脚本已经做了以下优化：

### 训练参数优化：
- `batch_size=16` (原来32)
- `epochs=15` (原来20)
- `max_samples_per_class=300` (原来500)

### 生成参数优化：
- `num_images_to_generate=8` (原来16)
- `num_inference_steps=20` (原来100)

### 进一步节省显存：
如果还是显存不够，可以进一步调整：

```bash
python validation/single_user_validation.py \
    --target_user_id 1 \
    --real_data_root "/kaggle/input/dataset" \
    --batch_size 8 \
    --max_samples_per_class 200 \
    --num_images_to_generate 4 \
    --epochs 10
```

## 📁 输出结构

```
user_1_validation/
├── user_01_classifier.pth       # 训练好的分类器
├── user_01_history.json         # 训练历史
├── user_01_training.png         # 训练曲线图
├── user_01_validation.json      # 验证结果
├── validation_report.md         # 验证报告
└── generated_images/            # 生成的图像 (如果选择生成)
    └── user_01/
        ├── generated_000.png
        ├── generated_001.png
        └── ...
```

## 📊 结果解读

### 验证结果示例：
```
📊 验证结果:
  图像数量: 8
  成功数量: 7
  成功率: 87.5%
  平均置信度: 0.856
  置信度范围: [0.234, 0.967]

🎉 优秀！生成图像很好地保持了用户 1 的特征
```

### 效果评估标准：
- **优秀 (≥80%)**：🎉 生成图像很好地保持了用户特征
- **良好 (≥60%)**：✅ 生成图像较好地保持了用户特征  
- **一般 (≥40%)**：⚠️ 生成图像部分保持了用户特征
- **较差 (<40%)**：❌ 生成图像未能很好保持用户特征

## 🔧 参数说明

### 必需参数：
- `--target_user_id`: 目标用户ID (如1, 5, 10等)
- `--real_data_root`: 真实数据根目录

### 训练参数：
- `--epochs`: 训练轮数 (推荐10-20)
- `--batch_size`: 批次大小 (显存不够可减少到8或4)
- `--max_samples_per_class`: 每类最大样本数 (可减少到100-200)

### 生成参数 (可选)：
- `--generate_images`: 是否生成图像
- `--vae_path`: VAE模型路径
- `--unet_path`: UNet模型路径
- `--condition_encoder_path`: 条件编码器路径
- `--num_images_to_generate`: 生成图像数量

### 验证参数：
- `--confidence_threshold`: 置信度阈值 (默认0.8)
- `--generated_images_dir`: 已有生成图像目录

## 💡 使用建议

### 1. 选择合适的用户：
- 选择数据量较多的用户 (>50张图像)
- 选择特征明显的用户

### 2. 显存管理：
- 如果显存不够，优先减少batch_size
- 其次减少样本数量和生成图像数量
- 可以分步骤执行 (先训练，再生成，最后验证)

### 3. 验证策略：
- 先用一个用户验证方法可行性
- 如果效果好，再扩展到多个用户
- 可以选择不同类型的用户进行对比

## 🔍 故障排除

### 常见问题：

1. **显存不足**：
   ```bash
   # 减少batch_size
   --batch_size 4
   
   # 减少样本数量
   --max_samples_per_class 100
   ```

2. **找不到用户数据**：
   ```
   确保数据目录结构：
   /kaggle/input/dataset/
   ├── user_01/
   ├── user_02/
   └── ...
   ```

3. **训练效果差**：
   - 检查数据质量和数量
   - 增加训练轮数
   - 调整学习率

4. **生成失败**：
   - 检查模型路径是否正确
   - 确认推理脚本是否正常工作

## 🎯 实际使用示例

### Kaggle环境完整示例：

```bash
# 1. 只验证用户1，节省显存
python validation/single_user_validation.py \
    --target_user_id 1 \
    --real_data_root "/kaggle/input/dataset" \
    --output_dir "/kaggle/working/user_1_validation" \
    --generate_images \
    --vae_path "/kaggle/input/final-model" \
    --unet_path "/kaggle/input/diffusion-final-model" \
    --condition_encoder_path "/kaggle/input/diffusion-final-model/condition_encoder.pt" \
    --epochs 15 \
    --batch_size 16 \
    --num_images_to_generate 8

# 2. 查看结果
cat /kaggle/working/user_1_validation/validation_report.md
```

这样你就可以用最少的资源验证生成模型的效果了！🎨
