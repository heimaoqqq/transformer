# 支持的目录格式

## 🎯 概述

验证系统现在支持多种用户数据目录格式，自动识别并适配不同的数据组织方式。

## 📁 支持的目录格式

### 真实数据目录格式：

系统会自动查找以下格式的用户目录：

1. **`user_01`, `user_02`, ...** (标准格式)
2. **`user_1`, `user_2`, ...** (简化格式)
3. **`ID_1`, `ID_2`, ...** (ID格式) ✅ **你的数据格式**
4. **`1`, `2`, ...** (纯数字格式)

### 生成图像目录格式：

生成图像目录也支持相同的格式：

1. **`user_01`, `user_02`, ...`**
2. **`user_1`, `user_2`, ...`**
3. **`ID_1`, `ID_2`, ...`**
4. **`1`, `2`, ...`**

## 🔍 自动识别逻辑

### 目标用户查找：
当指定 `--target_user_id 1` 时，系统会依次查找：
- `user_01/`
- `user_1/`
- `ID_1/` ✅ **匹配你的格式**
- `1/`

### 其他用户查找：
系统会自动识别所有符合格式的其他用户目录作为负样本。

## 📊 数据结构示例

### 你的数据结构：
```
/kaggle/input/dataset/
├── ID_1/           # 用户1的真实图像
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── ID_2/           # 用户2的真实图像
├── ID_3/
└── ...
```

### 生成图像结构：
```
/kaggle/working/generated_images/
├── ID_1/           # 用户1的生成图像
│   ├── generated_000.png
│   ├── generated_001.png
│   └── ...
├── ID_2/
└── ...
```

## 🚀 使用示例

### 针对你的数据格式：

```bash
# 单用户验证 (用户1)
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

### 多用户训练：

```bash
# 训练多个用户的分类器
python validation/train_user_classifiers.py \
    --real_data_root "/kaggle/input/dataset" \
    --user_ids 1 2 3 4 5 \
    --output_dir "/kaggle/working/user_classifiers" \
    --epochs 15 \
    --batch_size 16
```

## 🔧 调试信息

### 查找过程日志：
```
🔍 查找用户 1 的目录，支持格式: ['user_01', 'user_1', 'ID_1', '1']
✅ 找到目标用户目录: /kaggle/input/dataset/ID_1
📊 找到 5 个其他用户目录作为负样本
```

### 如果找不到用户：
```
❌ 未找到用户 1 的数据目录
```

**解决方案：**
1. 检查目录路径是否正确
2. 确认用户ID是否存在
3. 检查目录权限

## 💡 最佳实践

### 1. 数据组织建议：
- 保持一致的命名格式
- 确保每个用户目录包含足够的图像 (>20张)
- 图像格式统一 (PNG或JPG)

### 2. 目录结构验证：
```bash
# 检查数据目录结构
ls -la /kaggle/input/dataset/
# 应该看到 ID_1, ID_2, ID_3, ... 等目录

# 检查用户1的图像
ls -la /kaggle/input/dataset/ID_1/
# 应该看到图像文件
```

### 3. 常见问题排查：
- **路径错误**：确认 `--real_data_root` 指向正确的目录
- **权限问题**：确保目录可读
- **格式不匹配**：检查目录命名是否符合支持的格式

## 🎯 针对你的具体情况

### 数据路径：
- **真实数据根目录**：`/kaggle/input/dataset`
- **用户1数据目录**：`/kaggle/input/dataset/ID_1`
- **用户2数据目录**：`/kaggle/input/dataset/ID_2`
- **...**

### 推荐命令：
```bash
# 验证用户1
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

现在系统应该能够正确识别你的 `ID_1` 格式的数据目录了！🎨
