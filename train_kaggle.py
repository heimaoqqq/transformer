#!/usr/bin/env python3
"""
Kaggle环境专用训练脚本
一键运行VAE和扩散模型训练
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import argparse
import torch

# 导入配置
from kaggle_config import (
    setup_kaggle_environment, 
    get_kaggle_train_command,
    get_kaggle_generate_command,
    verify_kaggle_dataset,
    KAGGLE_CONFIG,
    OUTPUT_DIR
)

def check_gpu():
    """检查GPU状态"""
    print("🔍 Checking GPU status...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✅ GPU available: {gpu_name}")
        print(f"   GPU count: {gpu_count}")
        print(f"   GPU memory: {gpu_memory:.1f} GB")
        
        return True
    else:
        print("❌ No GPU available")
        return False

def run_command(command, description):
    """运行命令并处理输出"""
    print(f"\n🔄 {description}...")

    # 支持字符串和列表两种格式
    if isinstance(command, list):
        cmd_parts = command
        print(f"Command: {' '.join(command)}")
    else:
        # 将命令分割为列表
        cmd_parts = command.replace(" \\\n", " ").split()
        print(f"Command: {command}")

    try:
        # 运行命令
        process = subprocess.Popen(
            cmd_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出
        for line in process.stdout:
            print(line.rstrip())
        
        # 等待完成
        return_code = process.wait()
        
        if return_code == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed with return code {return_code}")
            return False
            
    except Exception as e:
        print(f"❌ {description} failed with error: {e}")
        return False

def train_vae(interactive=False):
    """训练VAE"""
    print("\n" + "="*50)
    print("🎯 Starting VAE Training")
    print("="*50)

    # 检查是否已有VAE模型
    vae_model_path = Path(OUTPUT_DIR) / "vae" / "final_model"
    if vae_model_path.exists():
        if interactive:
            try:
                response = input("VAE model already exists. Continue training? (y/n): ")
                if response.lower() != 'y':
                    print("Skipping VAE training")
                    return True
            except (EOFError, KeyboardInterrupt):
                print("⚠️  No input detected, using existing model...")
                return True
        else:
            print("⚠️  VAE model already exists. Using existing model...")
            print("   (Use --interactive flag to enable retraining option)")
            return True
    
    # 检查是否有多GPU，使用专用启动器
    if torch.cuda.device_count() > 1:
        print("🚀 检测到多GPU，使用专用启动器")
        command = ["python", "launch_multi_gpu.py", "vae"]
        success = run_command(command, "VAE Training (Multi-GPU)")
    else:
        # 单GPU使用原来的方式
        command = get_kaggle_train_command("vae")
        success = run_command(command, "VAE Training")
    
    if success:
        print("🎉 VAE training completed!")
        
        # 检查模型文件
        if vae_model_path.exists():
            print(f"✅ VAE model saved at: {vae_model_path}")
        else:
            print("⚠️  VAE model not found at expected location")
            return False
    
    return success

def train_diffusion(interactive=False):
    """训练扩散模型"""
    print("\n" + "="*50)
    print("🎯 Starting Diffusion Training")
    print("="*50)

    # 检查VAE模型是否存在
    vae_model_path = Path(OUTPUT_DIR) / "vae" / "final_model"
    if not vae_model_path.exists():
        print("❌ VAE model not found. Please train VAE first.")
        return False

    # 检查是否已有扩散模型
    diffusion_model_path = Path(OUTPUT_DIR) / "diffusion" / "final_model"
    if diffusion_model_path.exists():
        if interactive:
            try:
                response = input("Diffusion model already exists. Continue training? (y/n): ")
                if response.lower() != 'y':
                    print("Skipping diffusion training")
                    return True
            except (EOFError, KeyboardInterrupt):
                print("⚠️  No input detected, using existing model...")
                return True
        else:
            print("⚠️  Diffusion model already exists. Using existing model...")
            print("   (Use --interactive flag to enable retraining option)")
            return True
    
    # 检查是否有多GPU，使用专用启动器
    if torch.cuda.device_count() > 1:
        print("🚀 检测到多GPU，使用专用启动器")
        command = ["python", "launch_multi_gpu.py", "diffusion"]
        success = run_command(command, "Diffusion Training (Multi-GPU)")
    else:
        # 单GPU使用原来的方式
        command = get_kaggle_train_command("diffusion")
        success = run_command(command, "Diffusion Training")
    
    if success:
        print("🎉 Diffusion training completed!")
        
        # 检查模型文件
        unet_path = diffusion_model_path / "unet"
        encoder_path = diffusion_model_path / "condition_encoder.pt"
        
        if unet_path.exists() and encoder_path.exists():
            print(f"✅ Diffusion model saved at: {diffusion_model_path}")
        else:
            print("⚠️  Diffusion model files not found at expected location")
            return False
    
    return success

def generate_samples():
    """生成样本图像"""
    print("\n" + "="*50)
    print("🎯 Generating Sample Images")
    print("="*50)
    
    # 检查模型是否存在
    vae_model_path = Path(OUTPUT_DIR) / "vae" / "final_model"
    diffusion_model_path = Path(OUTPUT_DIR) / "diffusion" / "final_model"
    
    if not vae_model_path.exists():
        print("❌ VAE model not found")
        return False
    
    if not diffusion_model_path.exists():
        print("❌ Diffusion model not found")
        return False
    
    # 获取生成命令
    command = get_kaggle_generate_command()
    
    # 运行生成
    success = run_command(command, "Image Generation")
    
    if success:
        print("🎉 Image generation completed!")
        
        # 检查生成的图像
        output_path = Path(OUTPUT_DIR) / "generated_images"
        if output_path.exists():
            image_count = len(list(output_path.rglob("*.png")))
            print(f"✅ Generated {image_count} images at: {output_path}")
        else:
            print("⚠️  Generated images not found")
            return False
    
    return success

def estimate_training_time():
    """估算训练时间"""
    vae_config = KAGGLE_CONFIG["vae"]
    diffusion_config = KAGGLE_CONFIG["diffusion"]
    
    # 粗略估算 (基于经验值)
    vae_time = vae_config["num_epochs"] * 2  # 每个epoch约2分钟
    diffusion_time = diffusion_config["num_epochs"] * 3  # 每个epoch约3分钟
    
    total_time = vae_time + diffusion_time
    
    print(f"⏱️  Estimated training time:")
    print(f"   VAE: ~{vae_time//60}h {vae_time%60}m")
    print(f"   Diffusion: ~{diffusion_time//60}h {diffusion_time%60}m")
    print(f"   Total: ~{total_time//60}h {total_time%60}m")
    
    if total_time > 1800:  # 30小时
        print("⚠️  Warning: Estimated time exceeds Kaggle GPU limit (30h/week)")
        print("   Consider reducing epochs or using checkpoints")

def main():
    parser = argparse.ArgumentParser(description="Kaggle Training Pipeline")
    parser.add_argument("--stage", type=str, choices=[
        "setup", "vae", "diffusion", "generate", "all"
    ], default="all", help="Training stage to run")
    parser.add_argument("--skip_setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode (ask for confirmation)")
    
    args = parser.parse_args()
    
    print("🚀 Kaggle Micro-Doppler Training Pipeline")
    print("=" * 50)
    
    # 环境设置
    if not args.skip_setup:
        print("🔧 Setting up environment...")
        try:
            env_info = setup_kaggle_environment()
            print("✅ Environment setup completed")
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return
        
        # 验证数据集
        if not verify_kaggle_dataset():
            print("❌ Dataset verification failed")
            return
    
    # 检查GPU
    if not check_gpu():
        print("❌ GPU not available. Cannot proceed with training.")
        return
    
    # 估算训练时间
    if args.stage in ["all", "vae", "diffusion"]:
        estimate_training_time()

        # 根据模式决定是否需要确认
        if args.interactive:
            try:
                response = input("\nProceed with training? (y/n): ")
                if response.lower() != 'y':
                    print("Training cancelled")
                    return
            except (EOFError, KeyboardInterrupt):
                print("\n🚀 No input detected, starting training automatically...")
        else:
            print("\n🚀 Starting training automatically...")
            print("   (Use --interactive flag to enable confirmation prompts)")
            import time
            time.sleep(2)
    
    # 执行指定阶段
    success = True
    
    if args.stage == "setup":
        print("✅ Setup completed")
        
    elif args.stage == "vae":
        success = train_vae(interactive=args.interactive)

    elif args.stage == "diffusion":
        success = train_diffusion(interactive=args.interactive)

    elif args.stage == "generate":
        success = generate_samples()

    elif args.stage == "all":
        # 完整流程
        print("\n🎯 Running complete training pipeline...")

        # VAE训练
        if not train_vae(interactive=args.interactive):
            print("❌ VAE training failed")
            return

        # 扩散训练
        if not train_diffusion(interactive=args.interactive):
            print("❌ Diffusion training failed")
            return

        # 生成样本
        if not generate_samples():
            print("❌ Sample generation failed")
            return

        print("\n🎉 Complete pipeline finished successfully!")
    
    if success:
        print(f"\n✅ Stage '{args.stage}' completed successfully!")
    else:
        print(f"\n❌ Stage '{args.stage}' failed!")

if __name__ == "__main__":
    main()
