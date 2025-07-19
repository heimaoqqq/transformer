#!/usr/bin/env python3
"""
推理设置验证脚本
检查推理所需的所有文件和配置是否正确
"""

import os
import torch
from pathlib import Path
import json

def check_file_exists(path, description):
    """检查文件是否存在"""
    if Path(path).exists():
        size = Path(path).stat().st_size / (1024**2) if Path(path).is_file() else 0
        print(f"  ✅ {description}: {path} ({size:.1f} MB)")
        return True
    else:
        print(f"  ❌ {description}: {path} - 文件不存在!")
        return False

def check_directory_contents(path, description):
    """检查目录内容"""
    if Path(path).exists() and Path(path).is_dir():
        files = list(Path(path).iterdir())
        total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024**2)
        print(f"  ✅ {description}: {path} ({len(files)} 文件, {total_size:.1f} MB)")
        for file in files:
            print(f"    - {file.name}")
        return True
    else:
        print(f"  ❌ {description}: {path} - 目录不存在!")
        return False

def validate_model_configs():
    """验证模型配置文件"""
    print("\n🔍 验证模型配置...")
    
    # 检查VAE配置
    vae_config_path = "/kaggle/input/final-model/config.json"
    if Path(vae_config_path).exists():
        try:
            with open(vae_config_path, 'r') as f:
                vae_config = json.load(f)
            print(f"  ✅ VAE配置: {vae_config.get('_class_name', 'Unknown')}")
            print(f"    - 输入通道: {vae_config.get('in_channels', 'Unknown')}")
            print(f"    - 潜在通道: {vae_config.get('latent_channels', 'Unknown')}")
        except Exception as e:
            print(f"  ⚠️ VAE配置读取失败: {e}")
    
    # 检查UNet配置
    unet_config_path = "/kaggle/working/outputs/diffusion/final_model/unet/config.json"
    if Path(unet_config_path).exists():
        try:
            with open(unet_config_path, 'r') as f:
                unet_config = json.load(f)
            print(f"  ✅ UNet配置: {unet_config.get('_class_name', 'Unknown')}")
            print(f"    - 输入通道: {unet_config.get('in_channels', 'Unknown')}")
            print(f"    - 交叉注意力维度: {unet_config.get('cross_attention_dim', 'Unknown')}")
            print(f"    - Block输出通道: {unet_config.get('block_out_channels', 'Unknown')}")
        except Exception as e:
            print(f"  ⚠️ UNet配置读取失败: {e}")

def check_inference_requirements():
    """检查推理所需的所有文件"""
    print("🔍 检查推理所需文件...\n")
    
    all_good = True
    
    # 1. VAE模型
    print("1️⃣ VAE模型文件:")
    vae_base = "/kaggle/input/final-model"
    vae_files = [
        ("config.json", "VAE配置文件"),
        ("diffusion_pytorch_model.safetensors", "VAE权重文件(主)"),
        ("diffusion_pytorch_model.bin", "VAE权重文件(备用)")
    ]
    
    for file, desc in vae_files:
        path = f"{vae_base}/{file}"
        if not check_file_exists(path, desc):
            if file != "diffusion_pytorch_model.bin":  # bin文件是可选的
                all_good = False
    
    # 2. UNet模型
    print("\n2️⃣ UNet模型文件:")
    unet_base = "/kaggle/working/outputs/diffusion/final_model/unet"
    unet_files = [
        ("config.json", "UNet配置文件"),
        ("diffusion_pytorch_model.safetensors", "UNet权重文件(主)"),
        ("diffusion_pytorch_model.bin", "UNet权重文件(备用)")
    ]
    
    for file, desc in unet_files:
        path = f"{unet_base}/{file}"
        if not check_file_exists(path, desc):
            if file != "diffusion_pytorch_model.bin":  # bin文件是可选的
                all_good = False
    
    # 3. 条件编码器
    print("\n3️⃣ 条件编码器:")
    condition_path = "/kaggle/working/outputs/diffusion/final_model/condition_encoder.pt"
    if not check_file_exists(condition_path, "条件编码器权重"):
        all_good = False
    
    # 4. 检查采样图像 (可选，用于验证训练效果)
    print("\n4️⃣ 训练采样图像 (可选):")
    samples_dir = "/kaggle/working/outputs/diffusion/samples"
    check_directory_contents(samples_dir, "训练采样图像")
    
    return all_good

def generate_inference_command():
    """生成推理命令示例"""
    print("\n🚀 推理命令示例:")
    print("```bash")
    print("python inference/generate.py \\")
    print("    --vae_path \"/kaggle/input/final-model\" \\")
    print("    --unet_path \"/kaggle/working/outputs/diffusion/final_model/unet\" \\")
    print("    --condition_encoder_path \"/kaggle/working/outputs/diffusion/final_model/condition_encoder.pt\" \\")
    print("    --num_users 31 \\")
    print("    --user_ids 1 5 10 15 \\")
    print("    --num_images_per_user 3 \\")
    print("    --num_inference_steps 50 \\")
    print("    --guidance_scale 7.5 \\")
    print("    --output_dir \"/kaggle/working/generated_images\"")
    print("```")

def main():
    """主函数"""
    print("🧪 推理设置验证\n")
    print("=" * 50)
    
    # 检查基本环境
    print(f"🖥️  设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name()}")
        print(f"    内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 检查文件
    all_files_ready = check_inference_requirements()
    
    # 验证配置
    validate_model_configs()
    
    # 生成命令
    generate_inference_command()
    
    # 总结
    print("\n" + "=" * 50)
    if all_files_ready:
        print("🎉 所有推理文件准备就绪!")
        print("✅ 可以开始推理生成图像")
    else:
        print("⚠️ 部分文件缺失，请检查训练是否完成")
        print("❌ 推理可能无法正常运行")
    
    print("\n📋 推理文件清单:")
    print("  1. VAE模型: /kaggle/input/final-model/")
    print("  2. UNet模型: /kaggle/working/outputs/diffusion/final_model/unet/")
    print("  3. 条件编码器: /kaggle/working/outputs/diffusion/final_model/condition_encoder.pt")

if __name__ == "__main__":
    main()
