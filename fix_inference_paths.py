#!/usr/bin/env python3
"""
推理路径修复脚本
自动检测和修复推理所需的文件路径
"""

import os
from pathlib import Path
import argparse

def find_model_files():
    """查找所有可能的模型文件位置"""
    print("🔍 搜索模型文件...")
    
    # 可能的VAE路径
    vae_paths = [
        "/kaggle/input/final-model",
        "/kaggle/input/vae-model", 
        "/kaggle/input/vae",
        "/kaggle/working/outputs/vae_*/final_model"
    ]
    
    # 可能的扩散模型路径
    diffusion_paths = [
        "/kaggle/input/diffusion-final-model",
        "/kaggle/input/diffusion-model",
        "/kaggle/working/outputs/diffusion/final_model"
    ]
    
    found_files = {}
    
    # 查找VAE
    print("\n1️⃣ 查找VAE模型...")
    for path in vae_paths:
        if Path(path).exists():
            config_file = Path(path) / "config.json"
            weight_file = Path(path) / "diffusion_pytorch_model.safetensors"
            if config_file.exists() and weight_file.exists():
                found_files['vae'] = str(path)
                print(f"  ✅ 找到VAE: {path}")
                break
    else:
        print("  ❌ 未找到VAE模型")
    
    # 查找UNet
    print("\n2️⃣ 查找UNet模型...")
    for path in diffusion_paths:
        unet_path = Path(path) / "unet"
        if unet_path.exists():
            config_file = unet_path / "config.json"
            weight_file = unet_path / "diffusion_pytorch_model.safetensors"
            if config_file.exists() and weight_file.exists():
                found_files['unet'] = str(unet_path)
                print(f"  ✅ 找到UNet: {unet_path}")
                break
    else:
        print("  ❌ 未找到UNet模型")
    
    # 查找条件编码器
    print("\n3️⃣ 查找条件编码器...")
    for path in diffusion_paths:
        condition_file = Path(path) / "condition_encoder.pt"
        if condition_file.exists():
            found_files['condition_encoder'] = str(condition_file)
            print(f"  ✅ 找到条件编码器: {condition_file}")
            break
        # 也检查目录本身
        elif Path(path).exists():
            found_files['condition_encoder'] = str(path)  # 目录路径，脚本会自动处理
            print(f"  ✅ 找到条件编码器目录: {path}")
            break
    else:
        print("  ❌ 未找到条件编码器")
    
    return found_files

def generate_inference_command(found_files, user_ids=None, output_dir=None):
    """生成正确的推理命令"""
    if not all(key in found_files for key in ['vae', 'unet', 'condition_encoder']):
        print("❌ 缺少必要的模型文件，无法生成推理命令")
        return None
    
    user_ids = user_ids or [1, 5, 10, 15]
    output_dir = output_dir or "/kaggle/working/generated_images"
    
    command = f"""python inference/generate.py \\
    --vae_path "{found_files['vae']}" \\
    --unet_path "{found_files['unet']}" \\
    --condition_encoder_path "{found_files['condition_encoder']}" \\
    --num_users 31 \\
    --user_ids {' '.join(map(str, user_ids))} \\
    --num_images_per_user 3 \\
    --num_inference_steps 50 \\
    --guidance_scale 7.5 \\
    --output_dir "{output_dir}\""""
    
    return command

def create_inference_script(found_files, output_file="run_inference.sh"):
    """创建可执行的推理脚本"""
    command = generate_inference_command(found_files)
    if command is None:
        return False
    
    script_content = f"""#!/bin/bash
# 自动生成的推理脚本
# 生成时间: $(date)

echo "🚀 开始微多普勒图像生成..."
echo "📁 使用的模型文件:"
echo "  VAE: {found_files['vae']}"
echo "  UNet: {found_files['unet']}"
echo "  条件编码器: {found_files['condition_encoder']}"
echo ""

{command}

echo ""
echo "✅ 推理完成！"
echo "📁 生成的图像保存在: /kaggle/working/generated_images/"
"""
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    # 使脚本可执行
    os.chmod(output_file, 0o755)
    
    print(f"📝 推理脚本已保存: {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="修复推理路径并生成推理命令")
    parser.add_argument("--user_ids", type=int, nargs="+", default=[1, 5, 10, 15], help="要生成的用户ID")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/generated_images", help="输出目录")
    parser.add_argument("--create_script", action="store_true", help="创建可执行的推理脚本")
    
    args = parser.parse_args()
    
    print("🔧 推理路径修复工具")
    print("=" * 50)
    
    # 查找模型文件
    found_files = find_model_files()
    
    print("\n📋 找到的文件:")
    for key, path in found_files.items():
        print(f"  {key}: {path}")
    
    # 生成推理命令
    print("\n🚀 推理命令:")
    command = generate_inference_command(found_files, args.user_ids, args.output_dir)
    
    if command:
        print("```bash")
        print(command)
        print("```")
        
        # 创建脚本文件
        if args.create_script:
            create_inference_script(found_files)
    
    # 检查文件完整性
    print("\n🔍 文件完整性检查:")
    all_good = True
    
    for key, path in found_files.items():
        if Path(path).exists():
            if Path(path).is_file():
                size = Path(path).stat().st_size / (1024**2)
                print(f"  ✅ {key}: {size:.1f} MB")
            else:
                print(f"  ✅ {key}: 目录存在")
        else:
            print(f"  ❌ {key}: 文件不存在")
            all_good = False
    
    if all_good:
        print("\n🎉 所有文件准备就绪，可以开始推理！")
    else:
        print("\n⚠️ 部分文件缺失，请检查训练是否完成")

if __name__ == "__main__":
    main()
