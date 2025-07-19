# 推理代码修复说明

## 问题描述

在运行推理命令时遇到以下错误：
```
OSError: Error no file named scheduler_config.json found in directory /kaggle/input/diffusion-final-model.
```

## 问题原因

1. **调度器配置文件缺失**: 训练时没有保存调度器的配置文件 `scheduler_config.json`
2. **推理代码尝试从UNet路径加载调度器配置**: 代码使用 `DDIMScheduler.from_pretrained(unet_path, subfolder="scheduler")` 但该文件不存在

## 修复内容

### 1. 调度器创建修复

**修复前**:
```python
# 创建调度器
if scheduler_type == "ddim":
    self.scheduler = DDIMScheduler.from_pretrained(unet_path, subfolder="scheduler")
else:
    self.scheduler = DDPMScheduler.from_pretrained(unet_path, subfolder="scheduler")
```

**修复后**:
```python
# 创建调度器 - 使用与训练时相同的配置
if scheduler_type == "ddim":
    # 先创建DDPM调度器配置，然后转换为DDIM
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        variance_type="fixed_small",
        clip_sample=False,
        prediction_type="epsilon",
        thresholding=False,
        dynamic_thresholding_ratio=0.995,
        clip_sample_range=1.0,
        sample_max_value=1.0,
    )
    self.scheduler = DDIMScheduler.from_config(ddpm_scheduler.config)
else:
    self.scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        variance_type="fixed_small",
        clip_sample=False,
        prediction_type="epsilon",
        thresholding=False,
        dynamic_thresholding_ratio=0.995,
        clip_sample_range=1.0,
        sample_max_value=1.0,
    )
```

### 2. 用户ID验证优化

**修复前**:
```python
# 验证用户ID
for user_id in user_ids:
    if user_id < 0 or user_id >= self.num_users:
        raise ValueError(f"Invalid user_id {user_id}. Must be in range [0, {self.num_users-1}]")
```

**修复后**:
```python
# 验证用户ID - 考虑用户ID映射
for user_id in user_ids:
    # 获取实际的用户索引
    user_idx = self.user_id_mapping.get(user_id, user_id - 1 if user_id > 0 else 0)
    if user_idx < 0 or user_idx >= self.num_users:
        raise ValueError(f"Invalid user_id {user_id} (mapped to index {user_idx}). Index must be in range [0, {self.num_users-1}]")
```

## 修复效果

1. ✅ **解决调度器配置文件缺失问题**: 调度器现在直接使用代码配置，与训练时保持一致
2. ✅ **支持1-based用户ID**: 用户可以使用1, 5, 10, 15这样的用户ID，代码会自动转换为0-based索引
3. ✅ **保持训练时配置一致性**: 调度器参数与训练时完全相同
4. ✅ **更好的错误提示**: 用户ID验证错误时会显示映射后的索引
5. ✅ **自动设备检测**: 自动检测CUDA可用性，在CPU/GPU环境中都能正常工作

## 使用方法

现在可以正常运行推理命令：

```bash
python inference/generate.py \
    --vae_path "/kaggle/input/final-model" \
    --unet_path "/kaggle/input/diffusion-final-model" \
    --condition_encoder_path "/kaggle/input/diffusion-final-model/condition_encoder.pt" \
    --num_users 31 \
    --user_ids 1 5 10 15 \
    --num_images_per_user 16 \
    --num_inference_steps 100 \
    --guidance_scale 7.5 \
    --output_dir "/kaggle/working/generated_images"
```

## 参数说明

- `--vae_path`: VAE模型路径
- `--unet_path`: UNet模型路径  
- `--condition_encoder_path`: 条件编码器路径（可以是文件或包含condition_encoder.pt的目录）
- `--num_users`: 用户总数（31）
- `--user_ids`: 要生成的用户ID列表（支持1-based，如1 5 10 15）
- `--num_images_per_user`: 每个用户生成的图像数量
- `--num_inference_steps`: 推理步数（建议50-100）
- `--guidance_scale`: 引导强度（建议7.5）
- `--scheduler_type`: 调度器类型（"ddim"或"ddpm"，默认"ddim"）
- `--device`: 设备（"cuda"/"cpu"/"auto"，默认"auto"自动检测）
- `--output_dir`: 输出目录

## 用户ID映射

代码支持灵活的用户ID映射：
- **默认映射**: 用户ID N 映射到索引 N-1（如用户ID 1 -> 索引 0）
- **自定义映射**: 可以通过 `user_id_mapping` 参数提供自定义映射

## 注意事项

1. 确保所有模型文件路径正确
2. 用户ID应该在合理范围内（映射后的索引在[0, num_users-1]范围内）
3. 推理步数越多，生成质量越好，但耗时也越长
4. 引导强度影响生成图像与条件的匹配程度

## 文件结构

修复后的文件：
- `inference/generate.py`: 主要的推理脚本（已修复）
- `test_inference_fix.py`: 测试脚本（验证修复效果）
- `INFERENCE_FIX_README.md`: 本说明文件
