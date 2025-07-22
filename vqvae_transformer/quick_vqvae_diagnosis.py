#!/usr/bin/env python3
"""
快速VQ-VAE质量诊断
评估当前VQ-VAE在验证集上的真实性能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def load_vqvae_model(model_path):
    """加载VQ-VAE模型"""
    model_path = Path(model_path)
    
    # 尝试加载final_model
    final_model_path = model_path / "final_model"
    if final_model_path.exists():
        print(f"📂 加载final_model: {final_model_path}")
        try:
            from models.vqvae_model import MicroDopplerVQVAE
            model = MicroDopplerVQVAE.from_pretrained(final_model_path)
            print("✅ 成功加载final_model")
            return model
        except Exception as e:
            print(f"⚠️ final_model加载失败: {e}")
    
    # 尝试加载best_model.pth
    best_model_path = model_path / "best_model.pth"
    if best_model_path.exists():
        print(f"📂 加载best_model: {best_model_path}")
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu')
            from models.vqvae_model import MicroDopplerVQVAE
            
            model = MicroDopplerVQVAE(
                num_vq_embeddings=checkpoint['args'].codebook_size,
                commitment_cost=checkpoint['args'].commitment_cost,
                ema_decay=getattr(checkpoint['args'], 'ema_decay', 0.99),
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ 成功加载best_model")
            return model
        except Exception as e:
            print(f"⚠️ best_model加载失败: {e}")
    
    raise FileNotFoundError(f"未找到可用的VQ-VAE模型文件在 {model_path}")

def create_validation_dataloader(data_dir, batch_size=8):
    """创建验证数据加载器（使用与训练相同的分层划分）"""
    from data.micro_doppler_dataset import MicroDopplerDataset
    
    # 创建变换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 创建完整数据集
    full_dataset = MicroDopplerDataset(
        data_dir=data_dir,
        transform=transform,
        return_user_id=True,
    )
    
    # 使用相同的分层划分逻辑
    _, val_indices = stratified_split(full_dataset, train_ratio=0.8)
    
    # 创建验证集
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    print(f"📊 验证集信息:")
    print(f"   验证样本数: {len(val_dataset)}")
    print(f"   批次数量: {len(val_dataloader)}")
    
    return val_dataloader

def stratified_split(dataset, train_ratio=0.8):
    """分层划分（与训练代码保持一致）"""
    user_indices = {}
    for idx in range(len(dataset)):
        try:
            _, user_id = dataset[idx]
            user_id = user_id.item() if hasattr(user_id, 'item') else user_id
            
            if user_id not in user_indices:
                user_indices[user_id] = []
            user_indices[user_id].append(idx)
        except:
            continue
    
    train_indices = []
    val_indices = []
    
    import random
    random.seed(42)  # 与训练代码相同的种子
    
    for user_id, indices in user_indices.items():
        indices = indices.copy()
        random.shuffle(indices)
        
        user_train_size = max(1, int(len(indices) * train_ratio))
        
        if len(indices) == 1:
            train_indices.extend(indices)
        else:
            train_indices.extend(indices[:user_train_size])
            val_indices.extend(indices[user_train_size:])
    
    return train_indices, val_indices

def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    """简化的SSIM计算"""
    # 这里使用简化版本，实际应该使用专业的SSIM库
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    sigma1 = torch.var(img1)
    sigma2 = torch.var(img2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    return ssim.item()

def evaluate_vqvae(model, dataloader, device, max_batches=20):
    """评估VQ-VAE在验证集上的性能"""
    model.eval()
    model.to(device)
    
    total_psnr = 0
    total_ssim = 0
    total_vq_loss = 0
    total_recon_loss = 0
    num_samples = 0
    
    print(f"🔍 在验证集上评估VQ-VAE...")
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="评估中")):
            if batch_idx >= max_batches:  # 限制评估批次数量
                break
                
            images = images.to(device)
            
            # 前向传播
            outputs = model(images, return_dict=True)
            reconstructed = outputs.sample
            vq_loss = outputs.vq_loss
            
            # 计算重建损失
            recon_loss = torch.nn.functional.mse_loss(reconstructed, images)
            
            # 反归一化到[0,1]
            images_eval = (images * 0.5 + 0.5).clamp(0, 1)
            reconstructed_eval = (reconstructed * 0.5 + 0.5).clamp(0, 1)
            
            # 计算指标
            for i in range(images.size(0)):
                psnr = calculate_psnr(images_eval[i], reconstructed_eval[i])
                ssim = calculate_ssim(images_eval[i], reconstructed_eval[i])
                
                total_psnr += psnr
                total_ssim += ssim
                num_samples += 1
            
            total_vq_loss += vq_loss.item()
            total_recon_loss += recon_loss.item()
    
    # 计算平均指标
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_vq_loss = total_vq_loss / min(batch_idx + 1, max_batches)
    avg_recon_loss = total_recon_loss / min(batch_idx + 1, max_batches)
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'vq_loss': avg_vq_loss,
        'recon_loss': avg_recon_loss,
        'num_samples': num_samples
    }

def main():
    """主函数"""
    print("🔬 VQ-VAE快速质量诊断")
    print("=" * 50)
    
    # 配置
    vqvae_path = "/kaggle/input/best-model"  # 或者您的VQ-VAE模型路径
    data_dir = "/kaggle/input/dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 1. 加载模型
        print(f"📂 加载VQ-VAE模型...")
        model = load_vqvae_model(vqvae_path)
        
        # 2. 创建验证数据加载器
        print(f"📊 创建验证数据加载器...")
        val_dataloader = create_validation_dataloader(data_dir)
        
        # 3. 评估模型
        print(f"🎯 开始评估...")
        metrics = evaluate_vqvae(model, val_dataloader, device)
        
        # 4. 显示结果
        print(f"\n📊 VQ-VAE验证集性能:")
        print(f"   PSNR: {metrics['psnr']:.2f} dB")
        print(f"   SSIM: {metrics['ssim']:.4f}")
        print(f"   VQ损失: {metrics['vq_loss']:.4f}")
        print(f"   重建损失: {metrics['recon_loss']:.4f}")
        print(f"   评估样本数: {metrics['num_samples']}")
        
        # 5. 给出建议
        print(f"\n💡 诊断建议:")
        if metrics['psnr'] >= 25:
            print(f"✅ PSNR优秀 (≥25dB)，VQ-VAE质量很好，可以继续训练Transformer")
        elif metrics['psnr'] >= 20:
            print(f"⚠️ PSNR中等 (20-25dB)，VQ-VAE质量一般，建议先观察Transformer训练效果")
        else:
            print(f"❌ PSNR较低 (<20dB)，强烈建议重新训练VQ-VAE")
        
        if metrics['ssim'] >= 0.8:
            print(f"✅ SSIM优秀 (≥0.8)，结构相似性很好")
        elif metrics['ssim'] >= 0.6:
            print(f"⚠️ SSIM中等 (0.6-0.8)，结构相似性一般")
        else:
            print(f"❌ SSIM较低 (<0.6)，结构保持能力差")
            
    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        print(f"💡 建议：如果无法加载模型，可能需要重新训练VQ-VAE")

if __name__ == "__main__":
    main()
