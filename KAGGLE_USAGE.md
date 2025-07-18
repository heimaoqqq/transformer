# 🚀 Kaggle使用指南

## ⚡ 快速开始

### 1. 克隆仓库
```bash
!git clone git@github.com:heimaoqqq/VAE.git
%cd VAE
```

### 2. 修复环境
```bash
!python ultimate_fix_kaggle.py
```

### 3. 验证环境
```bash
!python check_vae_ldm_compatibility.py
```

### 4. 开始训练
```bash
# VAE训练
!python training/train_vae.py --data_dir "/kaggle/input/dataset" --resolution 128

# LDM训练  
!python training/train_diffusion.py --vae_path "outputs/vae/final_model" --resolution 128
```

## 🔧 故障排除

### ❌ 如果出现 `cached_download` 错误：

**错误信息**：
```
cannot import name 'cached_download' from 'huggingface_hub'
```

**解决方案**：

#### 方法1: 重启内核 (推荐)
1. 在Kaggle中：`Runtime → Restart Session`
2. 重新运行：`!python ultimate_fix_kaggle.py`

#### 方法2: 手动修复
```bash
!pip uninstall huggingface_hub diffusers transformers accelerate -y
!pip install huggingface_hub==0.16.4 diffusers==0.21.4 transformers==4.30.2 accelerate==0.20.3
```

#### 方法3: 验证修复
```bash
!python -c "from huggingface_hub import cached_download; print('✅ cached_download 可用')"
```

## 📋 关键版本组合

**稳定版本** (经过验证):
```
huggingface_hub==0.16.4  # 包含 cached_download
diffusers==0.21.4        # 兼容版本
transformers==4.30.2     # 稳定版本
accelerate==0.20.3       # 稳定版本
```

## 🎯 项目配置确认

### VAE配置：
- 输入分辨率: 128×128
- 潜在空间: 32×32×4
- 压缩比: 4倍
- 架构: 3层下采样

### LDM配置：
- UNet sample_size: 32
- 条件维度: 768
- 时间步: 1000
- 批次大小: 4

## ✅ 验证清单

运行以下命令确认环境正确：

```bash
# 1. 基础导入测试
!python -c "
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from huggingface_hub import cached_download
print('✅ 所有导入成功')
"

# 2. VAE测试
!python -c "
from diffusers import AutoencoderKL
import torch
vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, sample_size=128)
x = torch.randn(1, 3, 128, 128)
z = vae.encode(x).latent_dist.sample()
print(f'✅ VAE测试: {x.shape} → {z.shape}')
"

# 3. 完整兼容性测试
!python check_vae_ldm_compatibility.py
```

## 🚨 常见问题

### Q: 为什么需要特定版本？
A: `huggingface_hub >= 0.17.0` 移除了 `cached_download` 函数，但 `diffusers <= 0.24.0` 仍然依赖它。

### Q: 可以使用最新版本吗？
A: 不建议。最新版本可能有兼容性问题。建议使用经过验证的稳定版本组合。

### Q: 环境修复失败怎么办？
A: 
1. 重启Kaggle内核
2. 重新运行 `ultimate_fix_kaggle.py`
3. 如果仍然失败，手动安装指定版本

### Q: 训练时出现内存错误？
A: 
1. 减小批次大小：`--batch_size 2`
2. 使用梯度累积：`--gradient_accumulation_steps 2`
3. 启用混合精度：`--mixed_precision fp16`

## 🎉 成功标志

环境配置成功的标志：
- ✅ `cached_download` 导入无错误
- ✅ VAE 128×128→32×32 压缩正常
- ✅ UNet sample_size=32 匹配
- ✅ 完整训练工作流程通过

## 📞 获取帮助

如果仍有问题：
1. 检查Kaggle GPU设置是否启用
2. 确认数据集路径正确
3. 查看完整错误日志
4. 重启内核后重试

---

**记住：环境一致性是成功训练的关键！** 🎯
