# 环境配置说明

## 🎯 统一配置脚本

所有环境配置功能已整合到 `setup_kaggle_environment.py`：

### 功能特性
- ✅ **GPU优化配置**: 自动检测GPU类型并优化参数
- ✅ **依赖自动安装**: 按照diffusers官方要求安装
- ✅ **兼容性检查**: 测试所有关键组件
- ✅ **错误自动修复**: 多种备用安装方案
- ✅ **智能配置**: 根据硬件自动调整训练参数

## 📋 版本选择说明

### PyTorch版本策略
```python
# 优先级排序的安装方案
pytorch_options = [
    "torch==1.13.1 + CUDA 11.6",  # Kaggle兼容性最好
    "Kaggle预装版本",               # 通常已优化
    "torch==2.0.1 + CUDA 11.7",   # 较新版本
    "最新CUDA版本",                # 可能有兼容性问题
    "CPU版本",                     # 备用方案
]
```

### HuggingFace生态
- **完全按照diffusers 0.24.0官方要求**
- **使用hf_hub_download API** (不依赖cached_download)
- **自动处理版本兼容性**

### GPU智能配置
```python
# 根据GPU类型自动配置
gpu_configs = {
    "Tesla T4": {
        "batch_size": 16,
        "mixed_precision": True,
        "gradient_checkpointing": True
    },
    "Tesla P100": {
        "batch_size": 12,
        "mixed_precision": False,  # 不支持FP16
        "gradient_checkpointing": True
    },
    "Tesla V100": {
        "batch_size": 32,
        "mixed_precision": True,
        "gradient_checkpointing": False  # 内存充足
    }
}
```

## 🚀 使用方法

### Kaggle环境
```bash
# 克隆仓库
git clone https://github.com/heimaoqqq/VAE.git
cd VAE/vqvae_transformer

# 一键配置环境
python setup_kaggle_environment.py

# 验证安装
python check_environment.py

# 开始训练
python train_main.py --data_dir /kaggle/input/dataset
```

### 本地环境
```bash
cd vqvae_transformer

# 配置环境 (会自动适配本地硬件)
python setup_kaggle_environment.py

# 开始训练
python train_main.py --data_dir ./data
```

## 🔧 配置流程

### 自动执行步骤
1. **环境检查**: 检测Kaggle环境和GPU硬件
2. **清理环境**: 卸载冲突包，清理缓存
3. **安装PyTorch**: 多种方案确保成功
4. **安装HuggingFace**: 按官方要求安装
5. **安装其他依赖**: 数值计算、图像处理等
6. **测试验证**: 测试所有关键组件
7. **GPU配置**: 根据硬件优化参数

### 错误处理
- **自动重试**: 多种安装方案
- **降级策略**: 从新版本到稳定版本
- **备用方案**: GPU失败时使用CPU
- **详细日志**: 便于问题诊断

## 💡 技术要点

### CUDA兼容性
- **torch 1.13.1 + CUDA 11.6**: Kaggle环境兼容性最好
- **避免NCCL问题**: 使用经过验证的版本组合
- **自动降级**: 如果GPU有问题，自动使用CPU

### API兼容性
- **diffusers使用hf_hub_download**: 不依赖cached_download
- **版本范围控制**: 确保API稳定性
- **向后兼容**: 支持多种API版本

### 性能优化
- **混合精度训练**: 在支持的GPU上启用FP16
- **梯度检查点**: 减少内存使用
- **智能batch size**: 根据GPU内存自动调整

## 📊 预期效果

### 安装成功率
- **Kaggle环境**: >95% 成功率
- **本地环境**: >90% 成功率
- **错误恢复**: 自动尝试多种方案

### 性能提升
- **GPU训练**: 比CPU快5-10倍
- **内存优化**: 支持更大模型
- **稳定性**: 减少CUDA错误

## 🔍 故障排除

### 常见问题
1. **安装失败**: 重新运行脚本，会自动尝试其他方案
2. **GPU错误**: 脚本会自动降级到CPU版本
3. **版本冲突**: 脚本会自动清理并重装

### 手动修复
```bash
# 如果自动配置失败
pip uninstall torch transformers diffusers -y
pip cache purge
python setup_kaggle_environment.py

# 检查具体问题
python check_environment.py
```

## 📝 版本信息

### 推荐版本组合
```
numpy==1.26.4
torch==1.13.1 (CUDA 11.6)
torchvision==0.14.1
torchaudio==0.13.1
huggingface_hub>=0.19.4
transformers>=4.25.1
accelerate>=0.11.0
safetensors>=0.3.1
diffusers==0.24.0
```

### 更新日志
- **v1.0**: 整合所有环境配置功能
- **GPU优化**: 自动检测和配置
- **错误处理**: 多种备用方案
- **简化使用**: 一键配置所有环境
