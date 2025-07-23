#!/usr/bin/env python3
"""
内存优化的扩散模型训练脚本
专门针对16GB GPU内存限制优化
"""

import os
import torch
import gc

def setup_memory_optimization():
    """设置内存优化"""
    print("🔧 设置内存优化...")
    
    # 1. 设置CUDA内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 2. 启用内存映射
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 3. 设置内存增长策略
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 设置内存分片大小
        torch.cuda.set_per_process_memory_fraction(0.95)  # 使用95%的GPU内存
    
    print("✅ 内存优化设置完成")

def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def check_memory():
    """检查内存使用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU内存: {allocated:.2f}GB 已分配, {reserved:.2f}GB 已保留, {total:.2f}GB 总计")
        return allocated, reserved, total
    return 0, 0, 0

def main():
    """主函数"""
    print("🚀 内存优化的扩散模型训练")
    print("=" * 50)
    
    # 1. 设置内存优化
    setup_memory_optimization()
    
    # 2. 检查初始内存
    print("\n📊 初始内存状态:")
    check_memory()
    
    # 3. 清理内存
    clear_memory()
    
    # 4. 导入训练模块
    print("\n📦 导入训练模块...")
    try:
        import sys
        from pathlib import Path
        
        # 添加项目路径
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # 导入训练脚本
        from ..training.train_diffusion import main as train_main
        
        print("✅ 训练模块导入成功")
        
        # 5. 检查导入后内存
        print("\n📊 导入后内存状态:")
        check_memory()
        
        # 6. 开始训练
        print("\n🎯 开始训练...")
        train_main()
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        print("\n💡 内存优化建议:")
        print("1. 减小batch_size到1")
        print("2. 减少sample_interval")
        print("3. 使用梯度累积")
        print("4. 启用混合精度训练")
        
        # 显示内存状态
        print("\n📊 错误时内存状态:")
        check_memory()

if __name__ == "__main__":
    main()
