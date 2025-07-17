#!/usr/bin/env python3
"""
设置Accelerate配置用于Kaggle多GPU训练
"""

import os
import torch
import shutil
from pathlib import Path

def setup_accelerate_config():
    """设置Accelerate配置"""
    
    # 检测GPU数量
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"🎮 检测到 {gpu_count} 个GPU")
    
    if gpu_count <= 1:
        print("⚠️  单GPU环境，无需特殊配置")
        return
    
    # 创建Accelerate配置目录
    config_dir = Path.home() / ".cache" / "huggingface" / "accelerate"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成配置内容
    config_content = f"""compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: {gpu_count}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    
    # 写入配置文件
    config_file = config_dir / "default_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Accelerate配置已写入: {config_file}")
    print(f"🚀 配置为使用 {gpu_count} 个GPU")
    
    # 设置环境变量
    os.environ['ACCELERATE_CONFIG_FILE'] = str(config_file)
    
    return config_file

def test_accelerate_setup():
    """测试Accelerate设置"""
    try:
        from accelerate import Accelerator
        
        accelerator = Accelerator()
        
        print("\n🔍 Accelerate配置测试:")
        print(f"  设备: {accelerator.device}")
        print(f"  进程数: {accelerator.num_processes}")
        print(f"  分布式类型: {accelerator.distributed_type}")
        print(f"  是否主进程: {accelerator.is_main_process}")
        print(f"  本地进程索引: {accelerator.local_process_index}")
        
        if accelerator.num_processes > 1:
            print("✅ 多GPU配置成功!")
        else:
            print("⚠️  仍为单GPU模式")
            
    except Exception as e:
        print(f"❌ Accelerate测试失败: {e}")

def main():
    """主函数"""
    print("🚀 设置Kaggle多GPU训练环境")
    print("=" * 50)
    
    # 设置配置
    config_file = setup_accelerate_config()
    
    # 测试配置
    if config_file:
        test_accelerate_setup()
    
    print("\n💡 使用方法:")
    print("accelerate launch --config_file accelerate_config.yaml training/train_vae.py [args...]")
    print("或者直接运行: python train_kaggle.py --stage all")

if __name__ == "__main__":
    main()
