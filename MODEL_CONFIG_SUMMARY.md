# 🎯 模型配置总结

## 📊 **配置变更对比**

### **之前配置 (Stable Diffusion级别)**
```python
# UNet配置 - 工业级
block_out_channels=(320, 640, 1280, 1280)  # ~860M参数
cross_attention_dim=768
# 内存需求: ~15GB (仅模型权重)
# 适用: 数十亿图片数据集
```

### **新配置 (中型项目级别)**
```python
# UNet配置 - 中型项目
block_out_channels=(128, 256, 512, 512)    # ~200M参数
cross_attention_dim=512
# 内存需求: ~5GB (仅模型权重)
# 适用: 数万到数十万图片数据集
```

## 🔧 **完整模型配置**

### **VAE配置 (保持不变)**
```python
AutoencoderKL(
    in_channels=3,
    out_channels=3,
    down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
    up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
    block_out_channels=[128, 256, 512],  # 合理配置
    latent_channels=4,
    sample_size=128,
)
# 参数量: ~50M
# 内存需求: ~1GB
```

### **UNet配置 (已优化)**
```python
UNet2DConditionModel(
    sample_size=32,                        # 匹配VAE潜在空间
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 512),  # 中型配置
    down_block_types=(
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D", 
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D", 
        "CrossAttnUpBlock2D",
    ),
    cross_attention_dim=512,               # 与block_out_channels匹配
    attention_head_dim=8,
    use_linear_projection=True,
)
# 参数量: ~200M
# 内存需求: ~5GB
```

### **条件编码器配置 (已更新)**
```python
UserConditionEncoder(
    num_users=31,                          # 数据集用户数
    embed_dim=512,                         # 匹配UNet cross_attention_dim
    dropout=0.1
)
# 参数量: ~16K (31 * 512)
# 内存需求: 可忽略
```

## 📈 **性能对比**

| 配置 | 参数量 | 内存需求 | 训练时间 | 适用数据集 |
|------|--------|----------|----------|------------|
| **之前 (SD级别)** | ~860M | ~15GB | 很慢 | 数十亿图片 |
| **现在 (中型)** | ~200M | ~5GB | 适中 | 数万图片 |
| **轻量级选项** | ~50M | ~2GB | 快速 | 数千图片 |

## 🎯 **内存使用预估**

### **16GB GPU (Tesla P100)**
```
总内存: 16GB
├── 模型权重: ~5GB (UNet + VAE + 条件编码器)
├── 优化器状态: ~5GB (Adam优化器)
├── 梯度: ~5GB (反向传播)
├── 激活值: ~1GB (前向传播中间结果)
└── 可用缓冲: ~0GB
```

**结论**: batch_size=1 应该可以正常训练

### **训练参数建议**
```python
# 基础配置
--batch_size 1                    # 16GB GPU推荐
--gradient_accumulation_steps 4   # 模拟batch_size=4
--mixed_precision fp16            # 减少内存使用

# 内存优化
--sample_interval 1000            # 减少采样频率
--save_interval 5000              # 减少保存频率
```

## 🚀 **训练命令**

### **推荐训练命令**
```bash
python training/train_diffusion.py \
    --data_dir "/kaggle/input/dataset" \
    --vae_path "/kaggle/working/outputs/vae/final_model" \
    --resolution 128 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --cross_attention_dim 512 \
    --sample_interval 1000 \
    --output_dir "/kaggle/working/outputs/diffusion"
```

### **内存紧张时的配置**
```bash
# 如果仍然内存不足，可以进一步减小配置
python training/train_diffusion.py \
    --data_dir "/kaggle/input/dataset" \
    --vae_path "/kaggle/working/outputs/vae/final_model" \
    --resolution 128 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --cross_attention_dim 256 \
    --sample_interval 2000 \
    --output_dir "/kaggle/working/outputs/diffusion"
```

## ✅ **验证配置**

运行以下命令验证新配置：
```bash
# 1. 检查兼容性
python check_vae_ldm_compatibility.py

# 2. 诊断内存使用
python diagnose_memory_usage.py

# 3. 测试LDM配置
python test_ldm_config.py
```

## 📋 **配置文件位置**

已更新的文件：
- ✅ `training/train_diffusion.py` - 主训练脚本
- ✅ `test_ldm_config.py` - LDM配置测试
- ✅ `check_vae_ldm_compatibility.py` - 兼容性检查
- ✅ `verify_ldm_api.py` - API验证
- ✅ `diagnose_memory_usage.py` - 内存诊断

---

**现在配置已优化为适合16GB GPU的中型项目规模！** 🎉
