#!/usr/bin/env python3
"""
强制多GPU启动脚本
专门用于Kaggle环境的多GPU训练
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def setup_environment():
    """设置多GPU环境变量"""
    
    # 检测GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"🎮 检测到 {gpu_count} 个GPU")
    
    if gpu_count <= 1:
        print("⚠️  只有单GPU，无法启动多GPU训练")
        return False
    
    # 设置环境变量强制使用所有GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpu_count))
    os.environ['WORLD_SIZE'] = str(gpu_count)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    print(f"✅ 环境变量设置完成:")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"   WORLD_SIZE: {os.environ['WORLD_SIZE']}")
    
    return True

def launch_training(stage="vae"):
    """启动多GPU训练"""
    
    if not setup_environment():
        print("❌ 环境设置失败")
        return False
    
    gpu_count = torch.cuda.device_count()
    
    # 构建训练命令
    if stage == "vae":
        script_args = [
            "--data_dir", "/kaggle/input/dataset",
            "--output_dir", "/kaggle/working/outputs/vae",
            "--batch_size", "6",
            "--num_epochs", "40",
            "--learning_rate", "0.0001",
            "--mixed_precision", "fp16",
            "--gradient_accumulation_steps", "2",
            "--kl_weight", "1e-6",
            "--perceptual_weight", "0.05",
            "--freq_weight", "0.05",
            "--resolution", "256",
            "--num_workers", "2",
            "--save_interval", "10",
            "--log_interval", "5",
            "--sample_interval", "100",
            "--experiment_name", "kaggle_vae"
        ]
        script_path = "training/train_vae.py"
    
    elif stage == "diffusion":
        script_args = [
            "--data_dir", "/kaggle/input/dataset",
            "--vae_path", "/kaggle/working/outputs/vae/final_model",
            "--output_dir", "/kaggle/working/outputs/diffusion",
            "--batch_size", "4",
            "--num_epochs", "100",
            "--learning_rate", "0.0001",
            "--mixed_precision", "fp16",
            "--gradient_accumulation_steps", "4",
            "--cross_attention_dim", "768",
            "--num_train_timesteps", "1000",
            "--condition_dropout", "0.1",
            "--resolution", "256",
            "--val_split", "0.1",
            "--num_workers", "2",
            "--save_interval", "20",
            "--log_interval", "10",
            "--sample_interval", "100",
            "--val_interval", "50",
            "--experiment_name", "kaggle_diffusion"
        ]
        script_path = "training/train_diffusion.py"
    
    else:
        print(f"❌ 未知的训练阶段: {stage}")
        return False
    
    # 方法1: 使用accelerate launch
    cmd1 = [
        "accelerate", "launch",
        "--config_file", "accelerate_config.yaml",
        "--num_processes", str(gpu_count),
        script_path
    ] + script_args
    
    print(f"\n🚀 启动多GPU训练 ({stage}):")
    print("Command:", " ".join(cmd1))
    
    try:
        # 使用实时输出，不重定向
        result = subprocess.run(cmd1, check=True, text=True)
        print("✅ 训练完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ accelerate launch 失败: {e}")

        # 方法2: 使用torchrun (备用方案)
        print("\n🔄 尝试使用torchrun...")
        cmd2 = [
            "torchrun",
            "--nproc_per_node", str(gpu_count),
            "--master_port", "12355",
            script_path
        ] + script_args

        print("Command:", " ".join(cmd2))

        try:
            # 使用实时输出，不重定向
            result = subprocess.run(cmd2, check=True, text=True)
            print("✅ 训练完成 (torchrun)")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ torchrun 也失败: {e2}")

            # 方法3: 手动启动多进程 (最后方案)
            print("\n🔄 尝试手动多进程启动...")
            return launch_manual_multiprocess(script_path, script_args, gpu_count)
    
    return False

def launch_manual_multiprocess(script_path, script_args, gpu_count):
    """手动启动多进程训练"""
    
    processes = []
    
    for rank in range(gpu_count):
        env = os.environ.copy()
        env['LOCAL_RANK'] = str(rank)
        env['RANK'] = str(rank)
        env['CUDA_VISIBLE_DEVICES'] = str(rank)
        
        cmd = ["python", script_path] + script_args + [
            "--local_rank", str(rank)
        ]
        
        print(f"启动进程 {rank}: {' '.join(cmd)}")
        
        # 只重定向stderr，保持stdout实时输出
        process = subprocess.Popen(
            cmd,
            env=env,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(process)

    # 等待所有进程完成
    success = True
    for i, process in enumerate(processes):
        _, stderr = process.communicate()
        if process.returncode != 0:
            print(f"❌ 进程 {i} 失败:")
            if stderr:
                print(stderr)
            success = False
        else:
            print(f"✅ 进程 {i} 完成")
    
    return success

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python launch_multi_gpu.py <stage>")
        print("stage: vae 或 diffusion")
        sys.exit(1)
    
    stage = sys.argv[1]
    
    print("🚀 Kaggle多GPU训练启动器")
    print("=" * 50)
    
    success = launch_training(stage)
    
    if success:
        print(f"\n🎉 {stage} 训练完成!")
    else:
        print(f"\n❌ {stage} 训练失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
