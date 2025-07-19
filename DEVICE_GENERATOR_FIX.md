# 设备生成器修复

## 🐛 问题描述

在运行推理代码时遇到以下错误：
```
RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
```

## 🔍 问题原因

当使用CUDA设备时，PyTorch的随机数生成器（Generator）也必须在相同的设备上。之前的代码在设备自动检测后没有正确设置生成器的设备。

## 🔧 修复内容

**修复前：**
```python
# 设置随机种子
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)  # 默认在CPU上
else:
    generator = None
```

**修复后：**
```python
# 设置随机种子
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # 确保生成器在正确的设备上
    generator = torch.Generator(device=device).manual_seed(args.seed)
else:
    generator = None
```

## ✅ 修复效果

- ✅ 解决了CUDA设备上的生成器设备不匹配问题
- ✅ 确保随机数生成器与模型在同一设备上
- ✅ 保持随机种子的一致性和可重现性

## 🚀 使用方法

修复后可以正常运行：

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
    --device auto \
    --output_dir "/kaggle/working/generated_images"
```

## 📝 技术说明

PyTorch中的`torch.Generator`需要与使用它的张量在同一设备上：
- 当`device="cuda"`时，生成器也必须是CUDA类型
- 当`device="cpu"`时，生成器可以是CPU类型
- 使用`torch.Generator(device=device)`确保设备一致性

这个修复确保了在任何设备（CPU或CUDA）上都能正常工作。
