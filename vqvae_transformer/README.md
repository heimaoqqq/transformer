# 🎨 VQ-VAE + Transformer 微多普勒生成系统

基于diffusers和transformers的VQ-VAE + Transformer微多普勒时频图生成系统，专门针对小数据量和用户间微小差异优化。

## 🎯 方案优势

- ✅ **更低GPU要求**: 8GB即可训练，16GB绰绰有余
- ✅ **更适合小数据**: 离散化天然正则化，防止过拟合  
- ✅ **更好的条件控制**: Token级精确控制用户特征
- ✅ **防码本坍缩**: EMA更新、使用监控、自动重置
- ✅ **官方支持**: 基于diffusers和transformers，长期维护

## 📁 项目结构

```
vqvae_transformer/
├── models/                    # 模型定义
│   ├── __init__.py
│   ├── vqvae_model.py        # 防坍缩VQ-VAE模型
│   └── transformer_model.py  # 条件Transformer模型
├── training/                  # 训练脚本
│   ├── __init__.py
│   ├── train_vqvae.py        # VQ-VAE训练
│   └── train_transformer.py  # Transformer训练
├── inference/                 # 推理脚本
│   ├── __init__.py
│   └── generate.py           # 图像生成
├── validation/                # 验证框架
│   ├── __init__.py
│   └── validator.py          # 专用验证器
├── utils/                     # 工具函数
│   ├── __init__.py
│   ├── data_loader.py        # 数据加载器
│   └── metrics.py            # 评估指标
├── train_main.py             # 主训练脚本
├── generate_main.py          # 主生成脚本
├── validate_main.py          # 主验证脚本
├── requirements.txt          # 依赖管理
└── README.md                 # 本文件
```

## 🚀 快速开始

### 1. 环境安装 (重要！)

#### 一键安装 (推荐)
```bash
cd vqvae_transformer

# 统一环境安装脚本 - 自动检测环境并安装兼容版本
python setup_environment.py

# 验证环境是否正确
python check_environment.py
```

#### 手动安装 (如果自动安装失败)
```bash
cd vqvae_transformer

# ⚠️ 先卸载可能冲突的包
pip uninstall diffusers transformers huggingface-hub accelerate -y

# 安装兼容版本组合
pip install huggingface-hub==0.17.3 transformers==4.35.2 diffusers==0.24.0 accelerate==0.24.1
pip install -r requirements.txt

# 验证安装
python check_environment.py
```

#### 环境要求
- **Python**: 3.8+
- **CUDA**: 11.8 (推荐) 或 CPU
- **GPU内存**: 8GB+ (推荐16GB+)
- **系统内存**: 16GB+

### 2. 完整训练
```bash
python train_main.py \
    --data_dir "/kaggle/input/dataset" \
    --output_dir "/kaggle/working/outputs/vqvae_transformer" \
    --resolution 128 \
    --codebook_size 1024 \
    --num_users 31
```

### 3. 生成图像
```bash
python generate_main.py \
    --model_dir "/kaggle/working/outputs/vqvae_transformer" \
    --output_dir "generated_images" \
    --samples_per_user 10
```

### 4. 验证质量
```bash
python validate_main.py \
    --model_dir "/kaggle/working/outputs/vqvae_transformer" \
    --real_data_dir "/kaggle/input/dataset" \
    --generated_data_dir "generated_images" \
    --target_user_id 0
```

## 🔧 分阶段训练

如果需要分别训练两个阶段：

```bash
# 阶段1: 训练VQ-VAE
python training/train_vqvae.py \
    --data_dir "/kaggle/input/dataset" \
    --output_dir "outputs/vqvae" \
    --codebook_size 1024 \
    --batch_size 16

# 阶段2: 训练Transformer
python training/train_transformer.py \
    --data_dir "/kaggle/input/dataset" \
    --vqvae_path "outputs/vqvae" \
    --output_dir "outputs/transformer" \
    --batch_size 8
```

## 📊 核心技术

### VQ-VAE防坍缩机制
- **EMA更新**: 指数移动平均更新码本，避免梯度导致的坍缩
- **使用监控**: 实时监控码本使用率、熵值、活跃向量数
- **自动重置**: 智能重启未使用的码本向量
- **可视化**: 码本使用分布图表

### 条件Transformer
- **用户感知**: 专门的用户条件编码器
- **灵活架构**: 支持自注意力和交叉注意力
- **可控生成**: 温度、top-k、top-p采样控制
- **多样性增强**: 渐进式采样参数调整

## 📈 性能对比

| 特性 | 扩散模型 | VQ-VAE + Transformer |
|------|----------|---------------------|
| GPU内存需求 | ~15GB | ~8GB |
| 训练时间 | 长 | 中等 |
| 小数据适应性 | 差 | 优秀 |
| 条件控制精度 | 中等 | 优秀 |
| 生成速度 | 慢 | 快 |
| 用户特征保持 | 困难 | 容易 |

## 🎯 预期效果

- **码本利用率**: >80% (避免坍缩)
- **用户区分度**: 每用户20+独特码本向量  
- **生成质量**: 保持用户特征的同时增加多样性
- **训练效率**: 比扩散模型快2-3倍
- **内存友好**: 8GB GPU即可完整训练

## 📋 环境要求

- **Python**: 3.8+
- **PyTorch**: 2.1.0+
- **GPU**: 8GB+ VRAM
- **依赖**: 见 requirements.txt

## 🔍 故障排除

### 环境问题
1. **diffusers兼容性问题** (常见):
   ```bash
   # 如果遇到 "cannot import name 'cached_download'" 错误
   python setup_environment.py
   ```

2. **API不兼容**:
   ```bash
   # 重新安装兼容版本
   python setup_environment.py
   ```

3. **版本冲突**:
   ```bash
   # 检查版本
   python check_environment.py

   # 手动安装兼容版本
   pip uninstall diffusers transformers huggingface-hub -y
   pip install huggingface-hub==0.17.3 transformers==4.35.2 diffusers==0.24.0
   ```

3. **CUDA问题**:
   ```bash
   # 检查CUDA版本
   nvidia-smi

   # 安装对应CUDA版本的PyTorch
   # CUDA 11.8: torch==2.1.0+cu118
   # CUDA 12.1: torch==2.1.0+cu121
   # CPU only: torch==2.1.0
   ```

### 训练问题
1. **码本坍缩**: 检查EMA衰减率和commitment权重
2. **内存不足**: 降低batch_size或使用梯度累积
3. **生成质量差**: 调整采样温度和top-k/top-p参数
4. **用户特征不明显**: 增加用户条件编码器的复杂度

### 调试工具
- 码本使用可视化
- 训练损失曲线
- 生成样本质量监控
- 验证指标追踪

### 环境验证命令
```bash
# 快速检查
python check_environment.py

# 详细环境信息
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## 📄 许可证

本项目基于现有的Stable Diffusion项目，遵循相同的许可证。
