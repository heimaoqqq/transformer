#!/usr/bin/env python3
"""
GPU配置检测脚本
用于诊断Kaggle环境中的GPU配置问题
"""

import torch
import os
import subprocess
import sys

def check_nvidia_smi():
    """检查nvidia-smi输出"""
    print("🔍 检查nvidia-smi输出:")
    print("=" * 50)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("错误:", result.stderr)
    except FileNotFoundError:
        print("❌ nvidia-smi 未找到")
    except Exception as e:
        print(f"❌ 运行nvidia-smi失败: {e}")

def check_cuda_environment():
    """检查CUDA环境变量"""
    print("\n🔍 检查CUDA环境变量:")
    print("=" * 50)
    
    cuda_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_DEVICE_ORDER',
        'CUDA_LAUNCH_BLOCKING',
        'CUDA_CACHE_PATH',
        'CUDA_HOME',
        'CUDA_PATH'
    ]
    
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def check_pytorch_cuda():
    """检查PyTorch CUDA配置"""
    print("\n🔍 检查PyTorch CUDA配置:")
    print("=" * 50)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"\nGPU {i}:")
            print(f"  名称: {props.name}")
            print(f"  内存: {memory_gb:.1f} GB")
            print(f"  计算能力: {props.major}.{props.minor}")
            print(f"  多处理器数: {props.multi_processor_count}")
            
            # 检查GPU内存使用情况
            if torch.cuda.is_available():
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  已分配内存: {allocated:.2f} GB")
                print(f"  已保留内存: {reserved:.2f} GB")
                print(f"  可用内存: {memory_gb - reserved:.2f} GB")
    else:
        print("❌ CUDA不可用")

def check_accelerate_config():
    """检查Accelerate配置"""
    print("\n🔍 检查Accelerate配置:")
    print("=" * 50)
    
    try:
        from accelerate import Accelerator
        from accelerate.utils import gather_object
        
        accelerator = Accelerator()
        print(f"设备: {accelerator.device}")
        print(f"进程数: {accelerator.num_processes}")
        print(f"分布式类型: {accelerator.distributed_type}")
        print(f"是否主进程: {accelerator.is_main_process}")
        print(f"本地进程索引: {accelerator.local_process_index}")
        print(f"进程索引: {accelerator.process_index}")
        
    except ImportError:
        print("❌ Accelerate未安装")
    except Exception as e:
        print(f"❌ Accelerate检查失败: {e}")

def test_gpu_memory():
    """测试GPU内存分配"""
    print("\n🔍 测试GPU内存分配:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过内存测试")
        return
    
    for i in range(torch.cuda.device_count()):
        print(f"\n测试GPU {i}:")
        try:
            torch.cuda.set_device(i)
            
            # 尝试分配不同大小的内存
            test_sizes = [100, 500, 1000, 2000]  # MB
            
            for size_mb in test_sizes:
                try:
                    # 分配内存
                    size_bytes = size_mb * 1024 * 1024
                    tensor = torch.zeros(size_bytes // 4, device=f'cuda:{i}')  # float32 = 4 bytes
                    print(f"  ✅ 成功分配 {size_mb} MB")
                    del tensor
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    print(f"  ❌ 分配 {size_mb} MB 失败: {e}")
                    break
                    
        except Exception as e:
            print(f"  ❌ GPU {i} 测试失败: {e}")

def suggest_solutions():
    """建议解决方案"""
    print("\n💡 可能的解决方案:")
    print("=" * 50)
    
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if gpu_count == 0:
        print("1. 检查Kaggle Notebook设置中是否启用了GPU")
        print("2. 重启Notebook并重新选择GPU运行时")
        print("3. 检查Kaggle账户的GPU配额")
    elif gpu_count == 1:
        print("1. 检查Kaggle是否提供了双GPU环境")
        print("2. 可能需要申请更高级的GPU实例")
        print("3. 当前单GPU配置可能是正常的")
    else:
        print(f"✅ 检测到 {gpu_count} 个GPU，配置看起来正常")
        print("如果仍有内存问题，可能是:")
        print("1. 批次大小过大")
        print("2. 模型参数过多")
        print("3. 需要启用梯度检查点")

def main():
    """主函数"""
    print("🚀 GPU配置诊断工具")
    print("=" * 50)
    
    check_nvidia_smi()
    check_cuda_environment()
    check_pytorch_cuda()
    check_accelerate_config()
    test_gpu_memory()
    suggest_solutions()
    
    print("\n✅ 诊断完成!")

if __name__ == "__main__":
    main()
