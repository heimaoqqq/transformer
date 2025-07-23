# 🎨 VAE + 扩散模型 微多普勒生成系统

基于Stable Diffusion的VAE + 扩散模型微多普勒时频图生成系统，专门针对用户特征保持和高质量生成优化。

## 🎯 方案优势

- ✅ **高质量生成**: 基于Stable Diffusion架构，生成质量优秀
- ✅ **用户特征保持**: 条件扩散模型精确控制用户特征
- ✅ **成熟框架**: 基于diffusers库，API稳定可靠
- ✅ **灵活配置**: 支持多种VAE和扩散模型配置
- ✅ **完整验证**: 度量学习验证框架确保质量

## 📁 项目结构

```
vae_diffusion/
├── training/                  # 训练脚本
│   ├── __init__.py
│   ├── train_vae.py          # VAE训练核心
│   └── train_diffusion.py    # 条件扩散训练
├── inference/                 # 推理脚本
│   ├── __init__.py
│   └── generate.py           # 条件生成
├── validation/                # 验证框架
│   ├── __init__.py
│   ├── metric_learning_validator.py # 度量学习验证器
│   ├── statistical_validator.py     # 统计验证器
│   ├── user_classifier.py           # 用户分类器
│   └── validation_pipeline.py       # 验证流水线
├── utils/                     # 工具函数
│   ├── __init__.py
│   ├── data_loader.py        # 数据加载器
│   └── metrics.py            # 评估指标
├── train_celeba_standard.py  # 主训练脚本
├── train_diffusion_memory_optimized.py # 内存优化训练
├── train_improved_quality.py # 高质量训练
├── check_vae.py              # VAE质量检查
├── install_lpips.py          # 感知损失管理
├── ultimate_fix_kaggle.py    # 依赖修复工具
├── requirements.txt          # 依赖管理
└── README.md                 # 本文件
```

## 🚀 快速开始

### 1. 环境安装
```bash
cd vae_diffusion
pip install -r requirements.txt

# 安装感知损失 (可选)
python install_lpips.py
```

### 2. 标准训练
```bash
python train_celeba_standard.py \
    --data_dir "/kaggle/input/dataset" \
    --output_dir "/kaggle/working/outputs/vae_diffusion" \
    --resolution 128 \
    --num_users 31
```

### 3. 内存优化训练
```bash
python train_diffusion_memory_optimized.py \
    --data_dir "/kaggle/input/dataset" \
    --output_dir "/kaggle/working/outputs/vae_diffusion" \
    --resolution 128 \
    --batch_size 8
```

### 4. 高质量训练
```bash
python train_improved_quality.py \
    --data_dir "/kaggle/input/dataset" \
    --output_dir "/kaggle/working/outputs/vae_diffusion" \
    --resolution 128 \
    --use_ema \
    --use_lpips
```

### 5. 生成图像
```bash
python inference/generate.py \
    --model_dir "/kaggle/working/outputs/vae_diffusion" \
    --output_dir "generated_images" \
    --samples_per_user 10
```

## 🔧 核心技术

### VAE架构
- **编码器**: 将图像编码为潜在表示
- **解码器**: 从潜在表示重建图像
- **KL散度**: 正则化潜在空间分布
- **感知损失**: 提升重建质量 (可选)

### 扩散模型
- **前向过程**: 逐步添加噪声
- **反向过程**: 逐步去噪生成
- **条件控制**: 用户ID条件生成
- **分类器引导**: 增强条件控制 (可选)

### 验证框架
- **度量学习**: Siamese网络验证用户特征
- **统计验证**: FID、IS等生成质量指标
- **用户分类**: 验证用户特征保持度
- **完整流水线**: 自动化验证流程

## 📊 性能对比

| 特性 | VAE + 扩散模型 | VQ-VAE + Transformer |
|------|---------------|---------------------|
| 生成质量 | 优秀 | 良好 |
| 用户特征保持 | 优秀 | 良好 |
| 训练时间 | 长 | 中等 |
| GPU内存需求 | ~15GB | ~8GB |
| 模型复杂度 | 高 | 中等 |
| 生成速度 | 慢 (多步去噪) | 快 (一次前向) |

## 📋 环境要求

- **Python**: 3.8+
- **PyTorch**: 2.1.0+
- **GPU**: 15GB+ VRAM (推荐)
- **依赖**: 见 requirements.txt

## 🔍 故障排除

### 常见问题
1. **内存不足**: 
   ```bash
   # 使用内存优化版本
   python train_diffusion_memory_optimized.py
   ```

2. **生成质量差**: 
   ```bash
   # 使用高质量训练
   python train_improved_quality.py --use_ema --use_lpips
   ```

3. **用户特征不明显**: 
   ```bash
   # 增加条件权重
   python train_celeba_standard.py --condition_scale 2.0
   ```

4. **训练不稳定**: 
   ```bash
   # 检查VAE质量
   python check_vae.py --model_path /path/to/vae
   ```

### 调试工具
```bash
# VAE质量检查
python check_vae.py

# 兼容性检查
python check_diffusion_compatibility.py
python check_vae_ldm_compatibility.py

# 配置测试
python test_vae_config.py
python test_ldm_config.py
```

## 📈 训练策略

### 标准训练流程
1. **VAE预训练**: 学习图像重建
2. **扩散模型训练**: 学习条件生成
3. **联合微调**: 端到端优化 (可选)

### 内存优化策略
- **梯度累积**: 模拟大批次训练
- **混合精度**: 减少内存占用
- **检查点**: 节省激活内存

### 质量提升技巧
- **EMA**: 指数移动平均稳定训练
- **感知损失**: 提升视觉质量
- **分类器引导**: 增强条件控制

## 🎯 预期效果

- **生成质量**: FID < 50, IS > 2.0
- **用户特征保持**: 分类准确率 > 80%
- **训练稳定性**: 损失平滑下降
- **生成多样性**: 同用户内变化丰富

## 📄 许可证

本项目基于Stable Diffusion，遵循相应的开源许可证。
