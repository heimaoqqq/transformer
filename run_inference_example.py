#!/usr/bin/env python3
"""
推理代码使用示例
展示如何使用修复后的推理脚本
"""

import subprocess
import sys
from pathlib import Path

def run_inference_example():
    """运行推理示例"""
    print("🚀 微多普勒图像生成推理示例")
    print("=" * 50)
    
    # 示例参数 - 请根据实际情况修改路径
    example_params = {
        "vae_path": "/kaggle/input/final-model",
        "unet_path": "/kaggle/input/diffusion-final-model", 
        "condition_encoder_path": "/kaggle/input/diffusion-final-model/condition_encoder.pt",
        "num_users": 31,
        "user_ids": [1, 5, 10, 15],
        "num_images_per_user": 16,
        "num_inference_steps": 100,
        "guidance_scale": 7.5,
        "output_dir": "/kaggle/working/generated_images",
        "device": "auto",  # 自动检测设备
        "scheduler_type": "ddim",
        "seed": 42
    }
    
    print("📋 推理参数:")
    for key, value in example_params.items():
        if key == "user_ids":
            print(f"   {key}: {' '.join(map(str, value))}")
        else:
            print(f"   {key}: {value}")
    
    print("\n🔧 构建命令...")
    
    # 构建命令
    cmd = [
        "python", "inference/generate.py",
        "--vae_path", example_params["vae_path"],
        "--unet_path", example_params["unet_path"], 
        "--condition_encoder_path", example_params["condition_encoder_path"],
        "--num_users", str(example_params["num_users"]),
        "--user_ids"] + [str(uid) for uid in example_params["user_ids"]] + [
        "--num_images_per_user", str(example_params["num_images_per_user"]),
        "--num_inference_steps", str(example_params["num_inference_steps"]),
        "--guidance_scale", str(example_params["guidance_scale"]),
        "--scheduler_type", example_params["scheduler_type"],
        "--device", example_params["device"],
        "--output_dir", example_params["output_dir"],
        "--seed", str(example_params["seed"])
    ]
    
    print("💻 完整命令:")
    print(" ".join(cmd))
    
    print("\n📝 等效的bash命令:")
    bash_cmd = f"""python inference/generate.py \\
    --vae_path "{example_params['vae_path']}" \\
    --unet_path "{example_params['unet_path']}" \\
    --condition_encoder_path "{example_params['condition_encoder_path']}" \\
    --num_users {example_params['num_users']} \\
    --user_ids {' '.join(map(str, example_params['user_ids']))} \\
    --num_images_per_user {example_params['num_images_per_user']} \\
    --num_inference_steps {example_params['num_inference_steps']} \\
    --guidance_scale {example_params['guidance_scale']} \\
    --scheduler_type {example_params['scheduler_type']} \\
    --device {example_params['device']} \\
    --output_dir "{example_params['output_dir']}" \\
    --seed {example_params['seed']}"""
    
    print(bash_cmd)
    
    print("\n" + "=" * 50)
    print("📚 参数说明:")
    print("   • vae_path: VAE模型路径")
    print("   • unet_path: UNet扩散模型路径")
    print("   • condition_encoder_path: 用户条件编码器路径")
    print("   • num_users: 训练时的用户总数")
    print("   • user_ids: 要生成图像的用户ID列表（1-based）")
    print("   • num_images_per_user: 每个用户生成的图像数量")
    print("   • num_inference_steps: 扩散去噪步数（越多质量越好但越慢）")
    print("   • guidance_scale: 条件引导强度（7.5是经验值）")
    print("   • scheduler_type: 调度器类型（ddim更快，ddpm更准确）")
    print("   • device: 计算设备（auto自动检测，cuda使用GPU，cpu使用CPU）")
    print("   • output_dir: 生成图像保存目录")
    print("   • seed: 随机种子（确保结果可复现）")
    
    print("\n🎯 预期输出:")
    print("   生成的图像将保存在以下结构:")
    print("   /kaggle/working/generated_images/")
    print("   ├── user_01/")
    print("   │   ├── generated_000.png")
    print("   │   ├── generated_001.png")
    print("   │   └── ...")
    print("   ├── user_05/")
    print("   ├── user_10/")
    print("   └── user_15/")
    
    print("\n⚡ 性能提示:")
    print("   • 使用GPU可显著加速生成过程")
    print("   • 减少num_inference_steps可加快生成但可能影响质量")
    print("   • 批量生成多个用户比单独生成更高效")
    print("   • 建议先用较少的图像数量测试，确认无误后再大批量生成")
    
    return cmd

def run_interpolation_example():
    """运行插值示例"""
    print("\n🌈 用户插值生成示例")
    print("=" * 50)
    
    # 插值参数
    interpolation_params = {
        "vae_path": "/kaggle/input/final-model",
        "unet_path": "/kaggle/input/diffusion-final-model",
        "condition_encoder_path": "/kaggle/input/diffusion-final-model/condition_encoder.pt", 
        "num_users": 31,
        "interpolation_users": [1, 15],  # 在用户1和用户15之间插值
        "interpolation_steps": 10,
        "num_inference_steps": 50,
        "output_dir": "/kaggle/working/interpolation_images",
        "device": "auto",
        "seed": 42
    }
    
    print("📋 插值参数:")
    for key, value in interpolation_params.items():
        if key == "interpolation_users":
            print(f"   {key}: {' '.join(map(str, value))}")
        else:
            print(f"   {key}: {value}")
    
    # 构建插值命令
    cmd = [
        "python", "inference/generate.py",
        "--vae_path", interpolation_params["vae_path"],
        "--unet_path", interpolation_params["unet_path"],
        "--condition_encoder_path", interpolation_params["condition_encoder_path"],
        "--num_users", str(interpolation_params["num_users"]),
        "--interpolation",
        "--interpolation_users"] + [str(uid) for uid in interpolation_params["interpolation_users"]] + [
        "--interpolation_steps", str(interpolation_params["interpolation_steps"]),
        "--num_inference_steps", str(interpolation_params["num_inference_steps"]),
        "--device", interpolation_params["device"],
        "--output_dir", interpolation_params["output_dir"],
        "--seed", str(interpolation_params["seed"])
    ]
    
    print("\n💻 插值命令:")
    print(" ".join(cmd))
    
    print("\n📝 等效的bash命令:")
    bash_cmd = f"""python inference/generate.py \\
    --vae_path "{interpolation_params['vae_path']}" \\
    --unet_path "{interpolation_params['unet_path']}" \\
    --condition_encoder_path "{interpolation_params['condition_encoder_path']}" \\
    --num_users {interpolation_params['num_users']} \\
    --interpolation \\
    --interpolation_users {' '.join(map(str, interpolation_params['interpolation_users']))} \\
    --interpolation_steps {interpolation_params['interpolation_steps']} \\
    --num_inference_steps {interpolation_params['num_inference_steps']} \\
    --device {interpolation_params['device']} \\
    --output_dir "{interpolation_params['output_dir']}" \\
    --seed {interpolation_params['seed']}"""
    
    print(bash_cmd)
    
    print("\n🎯 插值输出:")
    print("   生成的插值图像将保存在:")
    print("   /kaggle/working/interpolation_images/")
    print("   └── interpolation_1_15/")
    print("       ├── step_000.png  (用户1)")
    print("       ├── step_001.png")
    print("       ├── ...")
    print("       ├── step_009.png  (用户15)")
    print("       └── combined.png  (拼接图)")
    
    return cmd

def main():
    """主函数"""
    print("🎨 微多普勒图像生成推理工具")
    print("基于修复后的推理代码")
    print("=" * 60)
    
    # 常规生成示例
    regular_cmd = run_inference_example()
    
    # 插值生成示例  
    interpolation_cmd = run_interpolation_example()
    
    print("\n" + "=" * 60)
    print("🚀 快速开始:")
    print("1. 确保模型文件路径正确")
    print("2. 根据需要修改参数")
    print("3. 复制上述命令到Kaggle notebook中运行")
    print("4. 检查生成的图像质量")
    
    print("\n🔧 故障排除:")
    print("• 如果遇到CUDA内存不足，尝试:")
    print("  - 减少num_images_per_user")
    print("  - 减少num_inference_steps") 
    print("  - 使用--device cpu")
    print("• 如果生成质量不佳，尝试:")
    print("  - 增加num_inference_steps")
    print("  - 调整guidance_scale")
    print("  - 检查模型文件是否正确")

if __name__ == "__main__":
    main()
