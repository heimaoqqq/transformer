# VQ-VAE + Transformer 微多普勒时频图生成项目

## 🎯 项目概述
基于diffusers VQModel和transformers GPT2的微多普勒时频图生成项目，支持用户特征条件控制。

## 🚀 快速开始

### 统一环境训练 (推荐)
```bash
# 1. 克隆项目
git clone git@github.com:heimaoqqq/transformer.git
cd transformer

# 2. 配置环境 (自动修复版本冲突)
python setup_unified_environment.py

# 3. 验证环境安装
python test_unified_environment.py

# 4. 完整API兼容性检查
python test_api_compatibility.py

# 5. 开始训练
python train_main.py --data_dir /path/to/dataset
```

### 分阶段训练 (备选)
```bash
# VQ-VAE阶段
python setup_vqvae_environment.py
python test_api_compatibility.py  # 验证VQ-VAE环境
python train_main.py --skip_transformer --data_dir /path/to/dataset

# Transformer阶段 (重启后)
python setup_transformer_environment.py
python test_api_compatibility.py  # 验证Transformer环境
python train_main.py --skip_vqvae --data_dir /path/to/dataset
```

## 📦 核心文件

### 环境配置
- `setup_unified_environment.py` - 统一环境配置 (推荐)
- `setup_vqvae_environment.py` - VQ-VAE专用环境
- `setup_transformer_environment.py` - Transformer专用环境

### 测试验证
- `test_api_compatibility.py` - **完整API兼容性检查** ⭐
- `test_unified_environment.py` - 统一环境测试
- `test_cross_environment_compatibility.py` - 跨环境兼容性测试

### 训练脚本
- `train_main.py` - 主训练脚本
- `training/train_vqvae.py` - VQ-VAE专用训练
- `training/train_transformer.py` - Transformer专用训练

### 模型定义
- `models/vqvae_model.py` - 自定义VQ-VAE模型
- `models/transformer_model.py` - 自定义Transformer模型

## 🔧 技术特点

### 环境管理
- ✅ **统一环境**: 使用diffusers官方配置
- ✅ **智能版本选择**: 自动选择最佳diffusers版本
- ✅ **完整API兼容性验证**: 全面的兼容性检查
- ✅ **分阶段备选**: 特殊情况下的解决方案

### 模型架构
- 🎨 **VQ-VAE**: 基于diffusers VQModel，支持图像离散化
- 🤖 **Transformer**: 基于transformers GPT2，支持序列生成
- 🎯 **条件控制**: 用户特征条件生成
- 💾 **跨环境兼容**: VQ-VAE模型支持跨环境使用

### 训练优化
- 🚀 **低GPU要求**: 8GB即可训练
- 📊 **小数据友好**: 离散化天然正则化
- ⚡ **灵活训练**: 支持完整训练和分阶段训练
- 🔄 **断点续训**: 支持训练中断恢复

## 🧪 API兼容性检查

### 完整检查功能
`test_api_compatibility.py` 提供全面的API兼容性验证：

#### 1. 模块版本检查
- PyTorch, diffusers, transformers版本验证
- HuggingFace生态系统组件检查

#### 2. diffusers API验证
- VQModel导入和实例化测试
- VectorQuantizer功能验证
- 构造函数参数兼容性检查

#### 3. transformers API验证
- GPT2模型导入和创建测试
- Tokenizer功能验证
- 配置参数兼容性检查

#### 4. 前向传播兼容性
- VQModel前向传播测试
- GPT2前向传播测试
- 自定义模型前向传播测试

#### 5. 保存/加载兼容性
- state_dict获取和加载测试
- 跨环境模型兼容性验证

#### 6. 详细报告生成
- 自动生成兼容性报告
- 警告信息捕获和分析
- 成功率统计和建议

### 使用方法
```bash
# 1. 配置环境 (自动修复版本冲突)
python setup_unified_environment.py

# 2. 运行完整API兼容性检查
python test_api_compatibility.py

# 3. 查看生成的报告
cat api_compatibility_report.txt
```

## 📋 系统要求
- Python 3.8+
- PyTorch 2.1.0+
- CUDA 12.1+ (推荐)
- GPU内存: 8GB+ (16GB推荐)

## 🎉 项目优势
1. **官方标准**: 遵循diffusers和transformers官方配置
2. **完整验证**: 全面的API兼容性检查系统
3. **简化部署**: 统一环境减少配置复杂度
4. **智能适配**: 自动选择最佳版本组合
5. **灵活训练**: 支持多种训练模式
6. **生产就绪**: 完整的错误处理和日志记录

## 📞 联系方式
- GitHub: https://github.com/heimaoqqq/transformer
- 项目地址: git@github.com:heimaoqqq/transformer.git

## 🔄 版本历史
- v1.0: 初始版本，分阶段训练
- v2.0: 统一环境配置，完整API兼容性检查

---

**🌟 特别推荐**: 使用 `test_api_compatibility.py` 进行完整的API兼容性检查，确保环境配置正确！
