# 🎯 VQ-VAE + Transformer 微多普勒时频图生成

基于VQ-VAE和Transformer的微多普勒时频图生成项目，支持**统一环境训练**和**分阶段训练**两种方式。

## 🎯 项目特点

- ✅ **统一环境训练**: 使用diffusers官方配置，一个环境支持全流程 (推荐)
- ✅ **分阶段训练**: 备选方案，解决特殊情况下的依赖问题
- ✅ **智能版本选择**: 自动选择最佳diffusers版本，确保VQModel可用
- ✅ **更低GPU要求**: 8GB即可训练，16GB绰绰有余
- ✅ **更适合小数据**: 离散化天然正则化，防止过拟合
- ✅ **更好的条件控制**: Token级精确控制用户特征
- ✅ **防码本坍缩**: EMA更新、使用监控、自动重置

## 📁 项目结构

```
vqvae_transformer/
├── models/                              # 模型定义
│   ├── vqvae_model.py                  # VQ-VAE模型 (MicroDopplerVQVAE)
│   └── transformer_model.py            # Transformer模型 (GPT2-based)
├── training/                            # 训练脚本
│   └── train_vqvae.py                  # VQ-VAE专用训练脚本
├── utils/                              # 工具函数
│   ├── data_loader.py                  # 数据加载器
│   └── metrics.py                      # 评估指标
├── validation/                         # 验证脚本
├── inference/                          # 推理脚本
├── setup_vqvae_environment.py          # 🔧 VQ-VAE阶段环境配置
├── setup_transformer_environment.py    # 🔧 Transformer阶段环境配置
├── test_cross_environment_compatibility.py # 🧪 跨环境兼容性测试
├── train_main.py                       # 主训练脚本 (支持分阶段)
├── generate_main.py                    # 生成脚本
└── requirements.txt                    # 基础依赖列表
## 🚀 快速开始

### 🎯 统一环境训练 (推荐方法)

#### **为什么推荐统一环境？**

基于diffusers官方配置：
- **官方支持**: `pip install diffusers[torch] transformers` 是官方推荐配置
- **简化部署**: 一个环境支持VQ-VAE和Transformer训练
- **减少冲突**: 避免环境切换带来的问题
- **智能版本**: 自动选择最佳diffusers版本，确保VQModel可用

#### **统一环境训练**

```bash
# 1. 配置统一环境 (自动修复版本冲突)
python setup_unified_environment.py

# 2. 验证环境安装
python test_unified_environment.py

# 3. 完整API兼容性检查
python test_api_compatibility.py

# 4. 完整训练流程
python train_main.py --data_dir /kaggle/input/dataset

# 或分步骤训练
python train_main.py --skip_transformer --data_dir /kaggle/input/dataset  # 仅VQ-VAE
python train_main.py --skip_vqvae --data_dir /kaggle/input/dataset        # 仅Transformer
```

### 🔄 分阶段训练 (备选方案)

如果遇到特殊的依赖问题，可以使用分阶段训练：

#### **阶段1: VQ-VAE训练**

```bash
# 1. 配置VQ-VAE专用环境
python setup_vqvae_environment.py

# 2. 验证VQ-VAE环境
python test_api_compatibility.py

# 3. 训练VQ-VAE (跳过Transformer)
python train_main.py --skip_transformer --data_dir /kaggle/input/dataset

# 或使用专用脚本
python training/train_vqvae.py --data_dir /kaggle/input/dataset --output_dir ./outputs/vqvae
```

**VQ-VAE环境特点**：
- ✅ 使用diffusers官方配置: `pip install diffusers[torch] transformers`
- ✅ `diffusers` 最新版本 (0.34.0，VQModel仍然可用)
- ✅ `transformers` 官方要求的依赖
- ✅ 正确导入路径: `from diffusers.models.autoencoders.vq_model import VQModel`
- ✅ 专注图像处理和编码/解码

#### **阶段2: Transformer训练**

```bash
# 1. 重启环境并配置Transformer专用环境
python setup_transformer_environment.py

# 2. 验证Transformer环境
python test_api_compatibility.py

# 3. 训练Transformer (跳过VQ-VAE)
python train_main.py --skip_vqvae --data_dir /kaggle/input/dataset

# 或使用专用脚本
python training/train_transformer.py --vqvae_path ./outputs/vqvae --data_dir /kaggle/input/dataset
```

**Transformer环境特点**：
- ✅ `transformers>=4.50.0` - 最新功能和性能
- ✅ `huggingface_hub>=0.30.0` - 最新API
- ✅ 专注序列生成和语言模型
- ✅ 加载保存的VQ-VAE模型权重 (完全兼容)

#### **跨环境兼容性测试**

```bash
# 测试VQ-VAE模型在不同环境间的兼容性
python test_cross_environment_compatibility.py
```

### 🔄 其他使用方法

#### **方法1: 同一Notebook分阶段**
```bash
# 阶段1: VQ-VAE训练
python setup_vqvae_environment.py
python train_main.py --skip_transformer --data_dir /kaggle/input/dataset

# 重启Notebook，阶段2: Transformer训练
python setup_transformer_environment.py
python train_main.py --skip_vqvae --data_dir /kaggle/input/dataset
```

#### **方法2: 两个独立Notebook**
```bash
# Notebook 1: VQ-VAE专用
python setup_vqvae_environment.py
python training/train_vqvae.py --output_dir /kaggle/working/vqvae_output

# Notebook 2: Transformer专用 (使用Kaggle Dataset共享VQ-VAE模型)
python setup_transformer_environment.py
python training/train_transformer.py --vqvae_path /kaggle/input/vqvae-model
```

#### **方法3: 传统统一环境 (可能有依赖冲突)**
```bash
# 安装基础依赖
pip install -r requirements.txt

# 完整训练 (可能遇到依赖冲突)
python train_main.py --data_dir /path/to/data
```

## 🎮 使用示例

### **生成图像 (需要Transformer环境)**
```bash
# 确保在Transformer环境中运行
python setup_transformer_environment.py

python generate_main.py \
    --model_dir "/kaggle/working/outputs/vqvae_transformer" \
    --output_dir "generated_images" \
    --samples_per_user 10
```

### **验证质量 (需要Transformer环境)**
```bash
# 确保在Transformer环境中运行
python setup_transformer_environment.py

python validate_main.py \
    --model_dir "/kaggle/working/outputs/vqvae_transformer" \
    --real_data_dir "/kaggle/input/dataset" \
    --generated_data_dir "generated_images" \
    --target_user_id 0
```

## 🔧 技术细节

### **依赖冲突问题**

| 组件 | VQ-VAE环境 | Transformer环境 |
|------|-----------|----------------|
| **diffusers** | 最新版本 (0.34.0) | 不需要 |
| **transformers** | 最新版本 (官方要求) | >=4.50.0 |
| **huggingface_hub** | 自动兼容版本 | >=0.30.0 |
| **PyTorch** | 2.1.0+cu121 | 2.1.0+cu121 |

### **跨环境兼容性保证**

1. **使用自定义模型类**: `MicroDopplerVQVAE` 继承但独立于diffusers
2. **PyTorch标准权重**: 保存`state_dict`而非整个对象
3. **配置参数保存**: 重建时使用保存的`args`
4. **接口稳定性**: Transformer只使用VQ-VAE的核心接口

### **环境要求**
- **Python**: 3.8+
- **CUDA**: 12.1 (推荐) 或 11.8
- **GPU内存**: 8GB+ (推荐16GB+)
- **系统内存**: 16GB+

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

#### **Q: 依赖冲突？**
```bash
# 使用分阶段训练
python setup_vqvae_environment.py    # VQ-VAE阶段
# 重启后
python setup_transformer_environment.py  # Transformer阶段
```

## 📋 环境使用指南

### **各脚本的环境要求**

| 脚本 | VQ-VAE环境 | Transformer环境 | 说明 |
|------|-----------|----------------|------|
| `train_main.py --skip_transformer` | ✅ | ❌ | VQ-VAE训练 |
| `training/train_vqvae.py` | ✅ | ❌ | VQ-VAE专用训练 |
| `train_main.py --skip_vqvae` | ❌ | ✅ | Transformer训练 |
| `training/train_transformer.py` | ❌ | ✅ | Transformer专用训练 |
| `generate_main.py` | ❌ | ✅ | 图像生成 (需要两个模型) |
| `validate_main.py` | ❌ | ✅ | 质量验证 (需要两个模型) |
| `test_cross_environment_compatibility.py` | ✅ | ✅ | 兼容性测试 |

### **推荐工作流程**

1. **VQ-VAE阶段** (在VQ-VAE环境):
   ```bash
   python setup_vqvae_environment.py
   python train_main.py --skip_transformer --data_dir /kaggle/input/dataset
   ```

2. **Transformer阶段** (在Transformer环境):
   ```bash
   python setup_transformer_environment.py
   python train_main.py --skip_vqvae --data_dir /kaggle/input/dataset
   ```

3. **生成和验证** (在Transformer环境):
   ```bash
   python generate_main.py --model_dir ./outputs/vqvae_transformer
   python validate_main.py --model_dir ./outputs/vqvae_transformer
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
1. **环境配置失败**:
   ```bash
   # 重新运行一键配置脚本
   python setup_kaggle_environment.py

   # 或手动清理后重装
   pip uninstall torch transformers diffusers -y
   pip cache purge
   python setup_kaggle_environment.py
   ```

2. **GPU兼容性问题**:
   ```bash
   # 如果遇到CUDA错误，脚本会自动尝试多种PyTorch版本
   # 包括CPU版本作为备用
   python setup_kaggle_environment.py
   ```

3. **导入错误**:
   ```bash
   # 如果遇到模块导入错误
   python check_environment.py  # 检查具体问题
   python setup_kaggle_environment.py  # 重新配置
   ```

### 推荐版本组合
```bash
# setup_kaggle_environment.py 使用的优化版本组合
numpy==1.26.4              # 兼容JAX
torch==1.13.1              # 与Kaggle CUDA兼容的稳定版本
torchvision==0.14.1        # 对应torch 1.13.1
torchaudio==0.13.1         # 对应torch 1.13.1
huggingface_hub>=0.19.4    # diffusers官方要求
transformers>=4.25.1       # diffusers官方要求
accelerate>=0.11.0         # diffusers官方要求
safetensors>=0.3.1         # diffusers官方要求
diffusers==0.24.0          # 目标版本

# 自动GPU配置:
# - Tesla T4: batch_size=16, 混合精度=True
# - Tesla P100: batch_size=12, 混合精度=False
# - Tesla V100: batch_size=32, 混合精度=True
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

### 依赖冲突解决
```bash
# 常见问题1: OfflineModeIsEnabled导入错误
# 错误: "cannot import name 'OfflineModeIsEnabled' from 'huggingface_hub.utils'"
# 解决: setup_unified_environment.py 已自动修复此问题

# 常见问题2: NumPy版本冲突
# 错误: "NumPy 1.x cannot be run in NumPy 2.x"
# 解决: 自动降级到NumPy 1.x版本

# 常见问题3: VQModel导入失败
# 使用分阶段训练作为备选
python setup_vqvae_environment.py
python test_api_compatibility.py
```

### 环境验证命令
```bash
# 完整API兼容性检查
python test_api_compatibility.py

# 快速环境检查
python test_unified_environment.py

# 详细环境信息
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## 📄 许可证

本项目基于现有的Stable Diffusion项目，遵循相同的许可证。
