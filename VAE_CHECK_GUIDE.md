# VAE检查工具使用指南

## 🔍 概述

`check_vae.py` 是一个集成的VAE检查工具，用于验证训练完成的VAE模型质量和重建效果。

## 🚀 快速开始

### 完整检查 (推荐)
```bash
python check_vae.py
```
自动执行所有检查：训练状态 → 模型加载 → 重建质量 → 总结建议

### 其他检查模式

#### 1. 仅检查训练状态
```bash
python check_vae.py --mode status
```

#### 2. 快速重建检查
```bash
python check_vae.py --mode quick --num_samples 8
```

#### 3. 潜在空间分析
```bash
python check_vae.py --mode latent
```

#### 4. 生成重建图像网格
```bash
# 生成8张重建对比图
python check_vae.py --mode generate --num_samples 8

# 生成并保存单独的对比图
python check_vae.py --mode generate --num_samples 8 --save_individual
```

#### 5. 简单左右对比 (推荐)
```bash
# 生成简单的左右对比图：左边原始，右边重建
python check_vae.py --mode simple --num_samples 8

# 生成更多样本
python check_vae.py --mode simple --num_samples 12
```

## 📊 检查内容

### 训练状态检查
- ✅ 检查输出目录和训练文件
- ✅ 验证模型配置和完整性
- ✅ 显示模型参数信息

### 模型加载测试
- ✅ 测试模型是否可以正常加载
- ✅ 验证前向传播功能
- ✅ 显示模型大小和压缩比

### 重建质量检查
- ✅ 生成原始vs重建对比图
- ✅ 计算质量指标 (MSE, PSNR, 相关系数)
- ✅ 显示差异热力图
- ✅ 给出质量评估和改进建议

### 潜在空间分析
- ✅ 分析潜在向量的统计特性
- ✅ 各通道的分布情况
- ✅ 验证编码器的有效性

### 重建图像生成
- ✅ 生成多张重建对比图
- ✅ 创建网格布局展示
- ✅ 保存单独的高质量对比图
- ✅ 包含差异热力图分析

## 📈 质量评估标准

| PSNR值 | 质量评估 | 相关系数 | 说明 |
|--------|----------|----------|------|
| > 25dB | 优秀 | > 0.9 | 重建质量很高，可以进行下一步 |
| 20-25dB | 良好 | 0.8-0.9 | 重建质量不错，基本可用 |
| 15-20dB | 一般 | 0.7-0.8 | 可接受的质量，建议优化 |
| < 15dB | 较差 | < 0.7 | 需要重新训练或调整参数 |

## 🛠️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `full` | 检查模式：status/quick/full/latent/generate/simple |
| `--output_dir` | `/kaggle/working/outputs` | 模型输出目录 |
| `--data_dir` | `/kaggle/input/dataset` | 测试数据目录 |
| `--num_samples` | `8` | 重建检查的样本数量 |
| `--save_individual` | `False` | 是否保存单独的重建对比图 |
| `--simple_layout` | `False` | 使用简单布局 (左原始右重建) |

## 📁 输出文件

- `vae_reconstruction_comparison.png` - 详细重建对比图 (3行布局)
- `vae_reconstruction_grid.png` - 网格重建对比图 (2行布局)
- `reconstruction_samples/` - 单独的高质量对比图文件夹
- `simple_comparisons/` - 简单左右对比图文件夹 ⭐**推荐**
- 控制台输出 - 详细的检查报告和建议

## 💡 常见问题

### Q: 提示"未找到可用模型"
**A:** 确保VAE训练已完成，检查 `/kaggle/working/outputs/` 目录下是否有 `final_model` 文件夹

### Q: 重建质量较差怎么办？
**A:** 尝试以下优化：
- 降低KL权重：`--kl_weight 1e-7`
- 延长训练时间：增加 `--num_epochs`
- 调整学习率：`--learning_rate 1e-4`

### Q: 显存不足
**A:** 减少检查样本数量：`--num_samples 4`

## 🔄 工作流程

1. **训练VAE模型**
   ```bash
   python train_celeba_standard.py
   ```

2. **检查模型质量**
   ```bash
   python check_vae.py
   ```

3. **根据结果决定下一步**
   - 质量良好 → 进行扩散模型训练
   - 质量一般 → 优化参数重新训练
   - 质量较差 → 检查数据和配置

## 🎯 CelebA标准配置

本工具专为CelebA标准配置优化：
- 输入分辨率：64×64
- 下采样：64→32→16→8
- 潜在空间：8×8×4
- 压缩比：48:1

## 📞 技术支持

如果遇到问题，请检查：
1. CUDA环境是否正常
2. 数据路径是否正确
3. 模型文件是否完整
4. 依赖包是否安装完整
