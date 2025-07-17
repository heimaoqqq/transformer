# Kaggle环境设置指南

## 🚀 从GitHub克隆项目到Kaggle

### 方法1: 使用HTTPS (推荐)

在Kaggle Notebook中运行：

```bash
# 1. 克隆项目
!git clone https://github.com/heimaoqqq/VAE.git

# 2. 进入项目目录
%cd VAE

# 3. 查看项目结构
!ls -la

# 4. 验证数据集
!python kaggle_config.py

# 5. 开始训练
!python train_kaggle.py --stage all
```

### 方法2: 使用SSH (需要配置密钥)

如果您在Kaggle中配置了SSH密钥：

```bash
# 1. 克隆项目
!git clone git@github.com:heimaoqqq/VAE.git

# 2. 进入项目目录
%cd VAE

# 3. 开始训练
!python train_kaggle.py --stage all
```

## 📋 完整的Kaggle训练流程

### 1. 创建Kaggle Notebook

1. 登录 [Kaggle](https://www.kaggle.com)
2. 点击 "Create" → "New Notebook"
3. 选择 "GPU" 加速器
4. 添加您的数据集 (路径: `/kaggle/input/dataset`)

### 2. 在Notebook中运行

```python
# Cell 1: 克隆项目
!git clone https://github.com/heimaoqqq/VAE.git
%cd VAE

# Cell 2: 验证环境和数据
!python kaggle_config.py

# Cell 3: 开始训练 (一键运行)
!python train_kaggle.py --stage all

# 或者分步运行:
# !python train_kaggle.py --stage vae      # VAE训练
# !python train_kaggle.py --stage diffusion # 扩散训练
# !python train_kaggle.py --stage generate  # 生成图像
```

### 3. 监控训练进度

```python
# 查看训练日志
!tail -f /kaggle/working/outputs/vae/training.log

# 查看生成的样本
from IPython.display import Image, display
import os

# 显示VAE重建样本
sample_dir = "/kaggle/working/outputs/vae/samples"
if os.path.exists(sample_dir):
    for img_file in sorted(os.listdir(sample_dir))[:5]:
        display(Image(os.path.join(sample_dir, img_file)))

# 显示扩散生成样本
sample_dir = "/kaggle/working/outputs/diffusion/samples"
if os.path.exists(sample_dir):
    for img_file in sorted(os.listdir(sample_dir))[:5]:
        display(Image(os.path.join(sample_dir, img_file)))
```

### 4. 保存结果

```python
# 压缩输出文件
!cd /kaggle/working && tar -czf micro_doppler_vae_results.tar.gz outputs/

# 下载结果 (在Kaggle界面中)
# 文件位置: /kaggle/working/micro_doppler_vae_results.tar.gz
```

## ⚙️ Kaggle环境配置

### GPU和内存设置

```python
# 检查GPU状态
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### 数据集路径验证

```python
# 验证数据集结构
import os
from pathlib import Path

data_dir = Path("/kaggle/input/dataset")
print(f"Dataset directory exists: {data_dir.exists()}")

if data_dir.exists():
    user_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('ID_')]
    print(f"Found {len(user_dirs)} user directories")
    
    for user_dir in sorted(user_dirs, key=lambda x: int(x.name.split('_')[1]))[:5]:
        images = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))
        print(f"{user_dir.name}: {len(images)} images")
```

## 🔧 故障排除

### 常见问题

1. **克隆失败**
   ```bash
   # 如果HTTPS克隆失败，尝试:
   !git config --global http.sslverify false
   !git clone https://github.com/heimaoqqq/VAE.git
   ```

2. **内存不足**
   ```python
   # 修改配置文件中的批次大小
   # 编辑 kaggle_config.py 中的 batch_size
   ```

3. **数据集路径错误**
   ```python
   # 检查数据集是否正确添加到Kaggle
   !ls -la /kaggle/input/
   ```

4. **训练中断**
   ```python
   # 从检查点恢复训练
   !python training/train_vae.py --resume_from_checkpoint /kaggle/working/outputs/vae/checkpoints/checkpoint_epoch_20.pt
   ```

### 性能优化

```python
# 如果训练太慢，可以调整配置:
# 1. 减少epoch数
# 2. 增加gradient_accumulation_steps
# 3. 使用更小的模型
```

## 📊 预期结果

### 训练时间
- **VAE训练**: 2-3小时
- **扩散训练**: 4-6小时
- **总计**: 6-9小时 (在Kaggle 30小时限制内)

### 输出文件
```
/kaggle/working/outputs/
├── vae/
│   ├── final_model/           # 训练好的VAE
│   └── samples/               # 重建样本
├── diffusion/
│   ├── final_model/           # 训练好的扩散模型
│   └── samples/               # 生成样本
└── generated_images/          # 最终生成的图像
    ├── ID_1/
    ├── ID_5/
    └── ...
```

### 质量指标
- VAE重建PSNR > 25 dB
- VAE重建SSIM > 0.8
- 生成图像具有明显的用户特征差异

## 🎯 成功标准

训练成功的标志：
- ✅ 无错误完成所有训练阶段
- ✅ 生成的图像看起来像真实的微多普勒时频图
- ✅ 不同用户生成的图像有明显差异
- ✅ 可以根据用户ID控制生成内容

## 📞 技术支持

如果遇到问题：
1. 检查Kaggle GPU配额是否充足
2. 确认数据集路径正确
3. 查看训练日志中的错误信息
4. 参考项目文档中的故障排除部分

祝您在Kaggle上训练顺利！🚀
