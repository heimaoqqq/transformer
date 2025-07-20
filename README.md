# 🎨 微多普勒时频图生成系统

基于深度学习的微多普勒时频图生成系统，提供两种不同的技术方案。

## 📁 项目结构

```
micro-doppler-generation/
├── vae_diffusion/             # VAE + 扩散模型方案
│   ├── training/             # 训练脚本
│   │   ├── train_vae.py     # VAE训练
│   │   └── train_diffusion.py # 扩散模型训练
│   ├── inference/            # 推理脚本
│   │   └── generate.py      # 条件生成
│   ├── validation/           # 验证框架
│   │   ├── metric_learning_validator.py # 度量学习验证
│   │   ├── statistical_validator.py     # 统计验证
│   │   ├── user_classifier.py           # 用户分类器
│   │   └── validation_pipeline.py       # 验证流水线
│   ├── utils/                # 工具函数
│   │   ├── data_loader.py   # 数据加载器
│   │   └── metrics.py       # 评估指标
│   ├── train_celeba_standard.py # 标准训练脚本
│   ├── train_diffusion_memory_optimized.py # 内存优化训练
│   ├── train_improved_quality.py # 高质量训练
│   ├── check_vae.py         # VAE质量检查
│   ├── requirements.txt     # 依赖管理
│   └── README.md            # VAE方案详细说明
├── vqvae_transformer/         # VQ-VAE + Transformer方案
│   ├── models/               # 模型定义
│   │   ├── vqvae_model.py   # 防坍缩VQ-VAE
│   │   └── transformer_model.py # 条件Transformer
│   ├── training/             # 训练脚本
│   │   ├── train_vqvae.py   # VQ-VAE训练
│   │   └── train_transformer.py # Transformer训练
│   ├── inference/            # 推理脚本
│   │   └── generate.py      # 图像生成
│   ├── validation/           # 验证框架
│   │   └── validator.py     # 专用验证器
│   ├── utils/                # 工具函数
│   │   ├── data_loader.py   # 数据加载器
│   │   └── metrics.py       # 评估指标
│   ├── train_main.py         # 主训练脚本
│   ├── generate_main.py      # 主生成脚本
│   ├── validate_main.py      # 主验证脚本
│   ├── setup_environment.py # 统一环境安装器
│   ├── check_environment.py # 环境检查器
│   ├── requirements.txt     # 独立依赖管理
│   └── README.md            # VQ-VAE方案详细说明
└── README.md                 # 本文件
```

## 🎯 两种方案对比

| 特性 | VAE + 扩散模型 | VQ-VAE + Transformer |
|------|---------------|---------------------|
| **生成质量** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **用户特征保持** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **训练速度** | ⭐⭐ | ⭐⭐⭐⭐ |
| **GPU内存需求** | 15GB+ | 8GB+ |
| **小数据适应性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **生成速度** | ⭐⭐ (多步去噪) | ⭐⭐⭐⭐⭐ (一次前向) |
| **技术成熟度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **环境兼容性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🚀 快速开始

### 方案1: VAE + 扩散模型 (高质量)
```bash
# 进入VAE项目目录
cd vae_diffusion

# 安装依赖
pip install -r requirements.txt

# 开始训练
python train_celeba_standard.py \
    --data_dir "/kaggle/input/dataset" \
    --output_dir "/kaggle/working/outputs/vae_diffusion" \
    --resolution 128 \
    --num_users 31
```

### 方案2: VQ-VAE + Transformer (高效率)
```bash
# 进入VQ-VAE项目目录
cd vqvae_transformer

# 一键安装环境 (重要!)
python setup_environment.py

# 验证环境
python check_environment.py

# 开始训练
python train_main.py \
    --data_dir "/kaggle/input/dataset" \
    --output_dir "/kaggle/working/outputs/vqvae_transformer" \
    --resolution 128 \
    --codebook_size 1024 \
    --num_users 31
```

## 📊 方案选择建议

### 🎯 选择VAE + 扩散模型，如果你：
- ✅ **追求最高质量**: 需要最佳的生成质量
- ✅ **GPU内存充足**: 有15GB+的GPU内存
- ✅ **数据量较大**: 每用户有1000+样本
- ✅ **时间充裕**: 可以接受较长的训练时间
- ✅ **技术成熟**: 希望使用成熟稳定的技术

### 🚀 选择VQ-VAE + Transformer，如果你：
- ✅ **GPU内存有限**: 只有8-16GB GPU内存
- ✅ **数据量较小**: 每用户少于500样本
- ✅ **快速迭代**: 需要快速训练和生成
- ✅ **精确控制**: 希望更好的用户特征控制
- ✅ **环境友好**: 需要更好的环境兼容性

## 🔧 环境要求

### 通用要求
- **Python**: 3.8+
- **PyTorch**: 2.1.0+
- **CUDA**: 11.8+ (推荐)

### VAE + 扩散模型
- **GPU内存**: 15GB+ (推荐P100/V100/A100)
- **训练时间**: 长 (数小时到数天)
- **依赖**: diffusers, transformers, accelerate

### VQ-VAE + Transformer  
- **GPU内存**: 8GB+ (T4即可)
- **训练时间**: 中等 (1-3小时)
- **依赖**: 自动环境管理，确保兼容性

## 🎨 生成效果预期

### VAE + 扩散模型
- **FID分数**: < 50
- **IS分数**: > 2.0
- **用户分类准确率**: > 85%
- **生成多样性**: 高

### VQ-VAE + Transformer
- **码本利用率**: > 80%
- **用户区分度**: 每用户20+独特向量
- **生成速度**: 比扩散模型快5-10倍
- **用户特征保持**: 优秀

## 🔍 故障排除

### 环境问题
```bash
# VQ-VAE项目环境问题
cd vqvae_transformer
python check_environment.py  # 检查环境
python setup_environment.py  # 重装环境

# VAE项目环境问题
cd vae_diffusion
python check_vae.py         # 检查VAE
python ultimate_fix_kaggle.py # 修复Kaggle环境
```

### 训练问题
- **内存不足**: 降低batch_size或使用梯度累积
- **生成质量差**: 调整学习率和训练轮数
- **用户特征不明显**: 增加条件权重或用户嵌入维度

## 📚 详细文档

- [VAE + 扩散模型详细说明](vae_diffusion/README.md)
- [VQ-VAE + Transformer详细说明](vqvae_transformer/README.md)

## 📄 许可证

本项目基于开源深度学习框架，遵循相应的许可证。

---

**🎯 推荐使用流程**:
1. 如果是新手或GPU内存有限，先尝试 **VQ-VAE + Transformer**
2. 如果追求最高质量且资源充足，使用 **VAE + 扩散模型**
3. 两种方案都可以并行开发和比较效果
