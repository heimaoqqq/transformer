# 🎯 分阶段训练指南

## 📋 概述

为了解决依赖冲突问题，我们将训练过程分为两个独立的阶段，每个阶段使用专门优化的环境：

- **阶段1**: VQ-VAE训练 (图像编码/解码)
- **阶段2**: Transformer训练 (序列生成)

## 🔧 环境配置策略

### **阶段1环境 (VQ-VAE专用)**
```python
# 专注于图像处理和VQ-VAE
diffusers==0.24.0          # VQ-VAE核心
huggingface_hub==0.25.2    # 支持cached_download
# 不安装transformers (避免冲突)
```

### **阶段2环境 (Transformer专用)**
```python
# 专注于序列生成和语言模型
transformers>=4.50.0       # 最新版本，最佳性能
huggingface_hub>=0.30.0    # 最新API
# 不需要diffusers (只使用保存的VQ-VAE模型)
```

## 🚀 使用方法

### **方法1: 在同一个Kaggle Notebook中分阶段运行**

#### **步骤1: VQ-VAE训练**
```bash
# 配置VQ-VAE环境
python setup_vqvae_environment.py

# 训练VQ-VAE (跳过Transformer)
python train_main.py --skip_transformer --data_dir /kaggle/input/dataset

# 或者直接使用VQ-VAE训练脚本
python training/train_vqvae.py --data_dir /kaggle/input/dataset --output_dir ./outputs/vqvae
```

#### **步骤2: 重启Notebook并配置Transformer环境**
```bash
# 重启Kaggle Notebook (清理环境)
# 重新克隆代码
git clone https://github.com/heimaoqqq/VAE.git
cd VAE/vqvae_transformer

# 配置Transformer环境
python setup_transformer_environment.py

# 训练Transformer (跳过VQ-VAE)
python train_main.py --skip_vqvae --data_dir /kaggle/input/dataset

# 或者直接使用Transformer训练脚本
python training/train_transformer.py --vqvae_path ./outputs/vqvae --data_dir /kaggle/input/dataset
```

### **方法2: 使用两个独立的Kaggle Notebook**

#### **Notebook 1: VQ-VAE训练**
```bash
git clone https://github.com/heimaoqqq/VAE.git
cd VAE/vqvae_transformer
python setup_vqvae_environment.py
python training/train_vqvae.py --data_dir /kaggle/input/dataset --output_dir /kaggle/working/vqvae_output
```

#### **Notebook 2: Transformer训练**
```bash
git clone https://github.com/heimaoqqq/VAE.git
cd VAE/vqvae_transformer
python setup_transformer_environment.py

# 从第一个notebook复制VQ-VAE模型
# 或者使用Kaggle Dataset功能共享模型

python training/train_transformer.py --vqvae_path /kaggle/input/vqvae-model --data_dir /kaggle/input/dataset
```

## 📊 优势对比

| 特性 | 统一环境 | 分阶段环境 |
|------|----------|-----------|
| 依赖冲突 | ❌ 有冲突 | ✅ 无冲突 |
| 版本选择 | ❌ 妥协版本 | ✅ 最优版本 |
| 功能完整性 | ❌ 受限 | ✅ 完整 |
| 配置复杂度 | ✅ 简单 | ⚠️ 中等 |
| 训练稳定性 | ❌ 不稳定 | ✅ 稳定 |

## 🎯 技术细节

### **VQ-VAE阶段依赖**
```python
# 核心依赖
diffusers==0.24.0          # VQModel, AutoencoderKL
huggingface_hub==0.25.2    # cached_download支持
torch>=2.0.0               # 深度学习框架

# 图像处理
opencv-python              # 图像预处理
pillow                     # 图像IO
matplotlib                 # 可视化
lpips                      # 感知损失

# 不需要transformers
```

### **Transformer阶段依赖**
```python
# 核心依赖
transformers>=4.50.0       # GPT2Config, GPT2LMHeadModel
huggingface_hub>=0.30.0    # 最新API
torch>=2.0.0               # 深度学习框架
accelerate>=0.25.0         # 训练加速

# 序列处理
tokenizers                 # 文本处理
einops                     # 张量操作

# 不需要diffusers (只加载保存的VQ-VAE模型权重)
```

## 🔍 故障排除

### **常见问题**

#### **Q: VQ-VAE模型在Transformer阶段找不到？**
```bash
# 确保模型路径正确
ls -la ./outputs/vqvae_transformer/vqvae/
# 应该看到 best_model.pth 或 final_model.pth

# 或者指定完整路径
python training/train_transformer.py --vqvae_path /kaggle/working/outputs/vqvae_transformer/vqvae
```

#### **Q: 环境配置失败？**
```bash
# 清理环境重试
pip cache purge
pip uninstall -y torch torchvision torchaudio transformers diffusers huggingface_hub accelerate
python setup_vqvae_environment.py  # 或 setup_transformer_environment.py
```

#### **Q: 内存不足？**
```bash
# 减小批次大小
python train_main.py --vqvae_batch_size 8 --transformer_batch_size 4
```

## 💡 最佳实践

1. **保存中间结果**: 每个阶段都保存完整的模型和日志
2. **使用Kaggle Dataset**: 在不同notebook间共享VQ-VAE模型
3. **监控资源使用**: 注意GPU内存和磁盘空间
4. **备份重要文件**: 定期保存训练进度

## 🎉 预期效果

使用分阶段环境后：
- ✅ **无依赖冲突**: 每个阶段使用最优版本
- ✅ **训练稳定**: 避免版本兼容性问题
- ✅ **功能完整**: 每个阶段都有完整功能
- ✅ **性能最优**: 使用最新版本的性能优化
