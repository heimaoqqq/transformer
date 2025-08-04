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
├── generate_with_guidance.py # 🎯 支持指导强度的生成脚本
├── validation_simple.py      # 🎯 简化条件验证脚本
├── example_validate_condition.py # 📖 验证使用示例
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
# 基础生成 (纯条件)
python inference/generate.py \
    --model_dir "/kaggle/working/outputs/vae_diffusion" \
    --output_dir "generated_images" \
    --samples_per_user 10

# 支持指导强度的生成 (推荐)
python generate_with_guidance.py \
    --vae_path "/kaggle/working/outputs/vae_diffusion/vae/final_model" \
    --unet_path "/kaggle/working/outputs/vae_diffusion/unet/final_model" \
    --condition_encoder_path "/kaggle/working/outputs/vae_diffusion/condition_encoder/final_model.pth" \
    --data_dir "/kaggle/input/dataset" \
    --user_ids 1 5 10 \
    --num_images_per_user 50 \
    --guidance_scale 1.5 \
    --num_inference_steps 50
```

### 6. 验证条件扩散效果 🎯
```bash
# 步骤1: 训练31个用户分类器
python validation_simple.py \
    --data_dir "/kaggle/input/dataset" \
    --action train \
    --output_dir "./validation_results" \
    --epochs 30 \
    --batch_size 32

# 步骤2: 交叉验证生成图像 (核心验证)
python validation_simple.py \
    --data_dir "/kaggle/input/dataset" \
    --action cross_validate \
    --generated_images_dir "generated_images/user_01" \
    --target_user_id 1 \
    --output_dir "./validation_results"

# 完整验证流程示例
python example_validate_condition.py
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
- **指导强度**: 支持CFG增强条件控制
  - `guidance_scale=1.0`: 纯条件生成
  - `guidance_scale=1.5-2.0`: 轻微CFG增强 (推荐)
  - `guidance_scale>3.0`: 强CFG增强

### 验证框架
- **用户分类器**: 31个ResNet-18二分类器验证用户特征
- **交叉验证**: 用所有分类器验证生成图像，确保条件控制有效
- **统计验证**: FID、IS等生成质量指标
- **度量学习**: Siamese网络验证用户特征 (可选)
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

## 🔍 条件扩散验证详解

### 验证原理
条件扩散模型的核心是**条件控制**：给定用户ID，生成该用户的特征图像。

**验证方法**：
1. **训练31个用户分类器** - 每个用户一个二分类器（是/不是该用户）
2. **交叉验证生成图像** - 用所有分类器验证生成图像
3. **评估条件控制效果** - 分析识别率和区分度

### 验证指标
- **目标用户识别率** > 70% - 目标用户分类器正确识别生成图像
- **其他用户拒绝率** > 70% - 其他用户分类器正确拒绝生成图像
- **区分度得分** > 0.3 - 目标识别率与其他拒绝率的差值
- **条件控制有效性** - 综合评估条件扩散是否真正有效

### 验证结果解读
```json
{
  "condition_effective": true,           // 条件控制是否有效
  "discrimination_score": 0.45,          // 区分度得分 (0.45 = 优秀)
  "target_user_performance": {
    "success_rate": 0.78,               // 目标用户识别率 78%
    "status": "good"
  },
  "other_users_performance": {
    "avg_success_rate": 0.23,           // 其他用户平均识别率 23%
    "status": "good"                    // 低识别率是好事(正确拒绝)
  }
}
```

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

3. **条件控制无效** (验证失败):
   ```bash
   # 检查条件编码器训练
   # 增加条件权重
   python train_celeba_standard.py --condition_scale 2.0

   # 重新验证
   python validation_simple.py --action cross_validate
   ```

4. **用户特征不明显**:
   ```bash
   # 增加条件权重
   python train_celeba_standard.py --condition_scale 2.0
   ```

5. **训练不稳定**:
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

### 生成质量指标
- **FID**: < 50 (生成图像与真实图像的分布距离)
- **IS**: > 2.0 (生成图像的质量和多样性)
- **PSNR**: > 25dB (重建质量)
- **SSIM**: > 0.8 (结构相似性)

### 条件控制效果 🎯
- **目标用户识别率**: > 70% (生成图像被正确用户分类器识别)
- **其他用户拒绝率**: > 70% (生成图像被其他分类器正确拒绝)
- **区分度得分**: > 0.3 (条件控制的有效性)
- **条件控制有效性**: true (综合评估通过)

### 训练稳定性
- **损失平滑下降**: VAE和扩散模型损失稳定收敛
- **生成多样性**: 同用户内变化丰富，不同用户间差异明显
- **用户特征保持**: 31个用户特征清晰可区分

## 📄 许可证

本项目基于Stable Diffusion，遵循相应的开源许可证。
