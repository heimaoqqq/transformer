#!/usr/bin/env python3
"""
Kaggle环境专用配置文件
针对您的数据集结构进行优化
"""

import os
from pathlib import Path

# 数据集配置
KAGGLE_DATA_DIR = "/kaggle/input/dataset"
OUTPUT_DIR = "/kaggle/working/outputs"
TEMP_DIR = "/kaggle/working/temp"

# 数据集信息
NUM_USERS = 31
USER_IDS = list(range(1, 32))  # 1到31
IMAGE_SIZE = 256

# 训练配置 (针对Kaggle环境优化)
KAGGLE_CONFIG = {
    # VAE训练配置 (内存优化)
    "vae": {
        "batch_size": 4,  # 减小批次大小避免OOM
        "num_epochs": 30,  # 减少epoch数以适应时间限制
        "learning_rate": 1e-4,
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 4,  # 增加梯度累积保持有效批次大小
        "kl_weight": 1e-6,
        "perceptual_weight": 0.0,  # 禁用感知损失节省内存
        "freq_weight": 0.05,
        "resolution": 128,  # 降低分辨率节省内存
        "num_workers": 1,  # 减少worker数
        "save_interval": 10,
        "log_interval": 5,
        "sample_interval": 200,  # 减少采样频率
    },
    
    # 扩散训练配置
    "diffusion": {
        "batch_size": 4,  # 更小的批次大小
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 4,
        "cross_attention_dim": 768,
        "num_train_timesteps": 1000,
        "condition_dropout": 0.1,
        "save_interval": 20,
        "log_interval": 10,
        "sample_interval": 200,
        "val_interval": 10,
    },
    
    # 数据配置
    "data": {
        "resolution": 256,
        "val_split": 0.2,
        "test_split": 0.1,
        "num_workers": 2,  # Kaggle环境限制
        "use_augmentation": False,  # 微多普勒对传统增强敏感，默认关闭
    },
    
    # 生成配置
    "generation": {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "scheduler_type": "ddim",
        "num_images_per_user": 5,
    }
}

def setup_kaggle_environment():
    """设置Kaggle环境"""
    print("🔧 Setting up Kaggle environment...")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 检查数据集
    data_path = Path(KAGGLE_DATA_DIR)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {KAGGLE_DATA_DIR}")
    
    # 验证用户目录
    missing_users = []
    for user_id in range(1, NUM_USERS + 1):
        user_dir = data_path / f"ID_{user_id}"
        if not user_dir.exists():
            missing_users.append(user_id)
    
    if missing_users:
        print(f"⚠️  Warning: Missing user directories: {missing_users}")
    else:
        print(f"✅ All {NUM_USERS} user directories found")
    
    # 统计图像数量
    total_images = 0
    user_image_counts = {}
    
    for user_id in range(1, NUM_USERS + 1):
        user_dir = data_path / f"ID_{user_id}"
        if user_dir.exists():
            images = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg")) + list(user_dir.glob("*.jpeg"))
            user_image_counts[user_id] = len(images)
            total_images += len(images)
    
    print(f"📊 Dataset statistics:")
    print(f"   Total images: {total_images}")
    print(f"   Average per user: {total_images / NUM_USERS:.1f}")
    print(f"   Min per user: {min(user_image_counts.values())}")
    print(f"   Max per user: {max(user_image_counts.values())}")
    
    return {
        "data_dir": KAGGLE_DATA_DIR,
        "output_dir": OUTPUT_DIR,
        "temp_dir": TEMP_DIR,
        "total_images": total_images,
        "user_image_counts": user_image_counts
    }

def get_kaggle_train_command(stage="vae"):
    """生成Kaggle训练命令"""
    if stage == "vae":
        config = KAGGLE_CONFIG["vae"]
        data_config = KAGGLE_CONFIG["data"]
        
        cmd = f"""python training/train_vae.py \\
    --data_dir {KAGGLE_DATA_DIR} \\
    --output_dir {OUTPUT_DIR}/vae \\
    --batch_size {config['batch_size']} \\
    --num_epochs {config['num_epochs']} \\
    --learning_rate {config['learning_rate']} \\
    --mixed_precision {config['mixed_precision']} \\
    --gradient_accumulation_steps {config['gradient_accumulation_steps']} \\
    --kl_weight {config['kl_weight']} \\
    --perceptual_weight {config['perceptual_weight']} \\
    --freq_weight {config['freq_weight']} \\
    --resolution {config['resolution']} \\
    --num_workers {config['num_workers']} \\
    --save_interval {config['save_interval']} \\
    --log_interval {config['log_interval']} \\
    --sample_interval {config['sample_interval']} \\
    --experiment_name kaggle_vae"""
        
        if data_config['use_augmentation']:
            cmd += " \\\n    --use_augmentation"
            
    elif stage == "diffusion":
        config = KAGGLE_CONFIG["diffusion"]
        data_config = KAGGLE_CONFIG["data"]
        
        cmd = f"""python training/train_diffusion.py \\
    --data_dir {KAGGLE_DATA_DIR} \\
    --vae_path {OUTPUT_DIR}/vae/final_model \\
    --output_dir {OUTPUT_DIR}/diffusion \\
    --batch_size {config['batch_size']} \\
    --num_epochs {config['num_epochs']} \\
    --learning_rate {config['learning_rate']} \\
    --mixed_precision {config['mixed_precision']} \\
    --gradient_accumulation_steps {config['gradient_accumulation_steps']} \\
    --cross_attention_dim {config['cross_attention_dim']} \\
    --num_train_timesteps {config['num_train_timesteps']} \\
    --condition_dropout {config['condition_dropout']} \\
    --resolution {data_config['resolution']} \\
    --val_split {data_config['val_split']} \\
    --num_workers {data_config['num_workers']} \\
    --save_interval {config['save_interval']} \\
    --log_interval {config['log_interval']} \\
    --sample_interval {config['sample_interval']} \\
    --val_interval {config['val_interval']} \\
    --experiment_name kaggle_diffusion"""
    
    return cmd

def get_kaggle_generate_command():
    """生成Kaggle推理命令"""
    config = KAGGLE_CONFIG["generation"]
    
    cmd = f"""python inference/generate.py \\
    --vae_path {OUTPUT_DIR}/vae/final_model \\
    --unet_path {OUTPUT_DIR}/diffusion/final_model/unet \\
    --condition_encoder_path {OUTPUT_DIR}/diffusion/final_model/condition_encoder.pt \\
    --num_users {NUM_USERS} \\
    --user_ids 1 5 10 15 20 25 31 \\
    --num_images_per_user {config['num_images_per_user']} \\
    --num_inference_steps {config['num_inference_steps']} \\
    --guidance_scale {config['guidance_scale']} \\
    --scheduler_type {config['scheduler_type']} \\
    --output_dir {OUTPUT_DIR}/generated_images"""
    
    return cmd

def print_kaggle_instructions():
    """打印Kaggle使用说明"""
    print("🚀 Kaggle环境使用说明")
    print("=" * 50)
    
    print("\n1. 环境设置:")
    print("   python kaggle_config.py")
    
    print("\n2. VAE训练:")
    print(get_kaggle_train_command("vae"))
    
    print("\n3. 扩散训练:")
    print(get_kaggle_train_command("diffusion"))
    
    print("\n4. 生成图像:")
    print(get_kaggle_generate_command())
    
    print("\n📋 注意事项:")
    print("- Kaggle GPU时间限制: 30小时/周")
    print("- 建议分阶段训练，保存检查点")
    print("- 使用混合精度训练节省内存")
    print("- 监控训练进度，及时调整参数")

def verify_kaggle_dataset():
    """验证Kaggle数据集"""
    print("🔍 Verifying Kaggle dataset...")
    
    try:
        from utils.data_loader import MicroDopplerDataset
        
        # 创建数据集
        dataset = MicroDopplerDataset(
            data_dir=KAGGLE_DATA_DIR,
            resolution=256,
            augment=False
        )
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Number of users: {dataset.num_users}")
        print(f"   User mapping: {dataset.user_to_idx}")
        
        # 测试加载一个样本
        sample = dataset[0]
        print(f"   Sample image shape: {sample['image'].shape}")
        print(f"   Sample user ID: {sample['user_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset verification failed: {e}")
        return False

if __name__ == "__main__":
    # 设置环境
    env_info = setup_kaggle_environment()
    
    # 验证数据集
    if verify_kaggle_dataset():
        print("\n🎉 Kaggle environment ready!")
        print_kaggle_instructions()
    else:
        print("\n❌ Please check your dataset structure")
