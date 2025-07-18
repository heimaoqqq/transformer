#!/usr/bin/env python3
"""
GPU内存使用诊断工具
找出内存占用的具体原因
"""

import torch
import gc
import psutil
import os

def check_system_memory():
    """检查系统内存"""
    print("💻 系统内存状态:")
    memory = psutil.virtual_memory()
    print(f"   总内存: {memory.total / 1024**3:.2f} GB")
    print(f"   已使用: {memory.used / 1024**3:.2f} GB ({memory.percent:.1f}%)")
    print(f"   可用: {memory.available / 1024**3:.2f} GB")

def check_gpu_memory():
    """检查GPU内存"""
    print("\n🎮 GPU内存状态:")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3
            
            print(f"   GPU {i} ({props.name}):")
            print(f"      总内存: {total:.2f} GB")
            print(f"      已分配: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
            print(f"      已保留: {reserved:.2f} GB ({reserved/total*100:.1f}%)")
            print(f"      可用: {total-reserved:.2f} GB")
    else:
        print("   ❌ CUDA不可用")

def check_process_memory():
    """检查当前进程内存"""
    print("\n🔍 当前进程内存:")
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"   RSS: {memory_info.rss / 1024**3:.2f} GB")
    print(f"   VMS: {memory_info.vms / 1024**3:.2f} GB")

def test_model_loading():
    """测试模型加载的内存使用"""
    print("\n🧪 测试模型加载内存使用:")
    
    # 清理初始内存
    gc.collect()
    torch.cuda.empty_cache()
    
    print("1️⃣ 初始状态:")
    check_gpu_memory()
    
    try:
        # 测试VAE加载
        print("\n2️⃣ 加载VAE...")
        from diffusers import AutoencoderKL
        
        # 模拟VAE配置
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[128, 256, 512],
            latent_channels=4,
            sample_size=128,
        )
        
        if torch.cuda.is_available():
            vae = vae.cuda()
        
        check_gpu_memory()
        
        # 测试UNet加载
        print("\n3️⃣ 加载UNet...")
        from diffusers import UNet2DConditionModel
        
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),  # 中型配置
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=512,  # 与中型配置匹配
            attention_head_dim=8,
            use_linear_projection=True,
        )
        
        if torch.cuda.is_available():
            unet = unet.cuda()
        
        check_gpu_memory()
        
        # 测试前向传播
        print("\n4️⃣ 测试前向传播...")
        with torch.no_grad():
            # 模拟输入
            if torch.cuda.is_available():
                test_input = torch.randn(1, 3, 128, 128).cuda()
                
                # VAE编码
                latents = vae.encode(test_input).latent_dist.sample()
                print(f"   VAE编码后: {latents.shape}")
                check_gpu_memory()
                
                # UNet前向
                timesteps = torch.randint(0, 1000, (1,)).cuda()
                conditions = torch.randn(1, 1, 512).cuda()  # 匹配新的cross_attention_dim
                
                noise_pred = unet(latents, timesteps, encoder_hidden_states=conditions, return_dict=False)[0]
                print(f"   UNet预测后: {noise_pred.shape}")
                check_gpu_memory()
                
                # VAE解码
                reconstructed = vae.decode(latents).sample
                print(f"   VAE解码后: {reconstructed.shape}")
                check_gpu_memory()
        
        # 清理测试
        print("\n5️⃣ 清理测试模型...")
        del vae, unet
        if 'latents' in locals():
            del latents, noise_pred, reconstructed, test_input
        gc.collect()
        torch.cuda.empty_cache()
        check_gpu_memory()
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        check_gpu_memory()

def analyze_memory_usage():
    """分析内存使用模式"""
    print("\n📊 内存使用分析:")
    
    if torch.cuda.is_available():
        # 获取详细的内存统计
        memory_stats = torch.cuda.memory_stats()
        
        print("详细内存统计:")
        for key, value in memory_stats.items():
            if 'bytes' in key:
                print(f"   {key}: {value / 1024**3:.2f} GB")
            else:
                print(f"   {key}: {value}")

def suggest_optimizations():
    """建议优化方案"""
    print("\n💡 内存优化建议:")
    
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"基于 {total_memory:.1f}GB GPU 的优化建议:")
        
        if total_memory < 12:
            print("   🔴 低内存GPU (<12GB):")
            print("      - batch_size = 1")
            print("      - 启用梯度检查点")
            print("      - 使用CPU卸载")
            print("      - 减少采样步数")
        elif total_memory < 20:
            print("   🟡 中等内存GPU (12-20GB):")
            print("      - batch_size = 1-2")
            print("      - 启用混合精度")
            print("      - 定期清理内存")
        else:
            print("   🟢 高内存GPU (>20GB):")
            print("      - batch_size = 2-4")
            print("      - 正常训练配置")
    
    print("\n🔧 通用优化:")
    print("   - 设置 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
    print("   - 使用 torch.cuda.empty_cache() 定期清理")
    print("   - 启用 torch.backends.cudnn.benchmark = True")
    print("   - 减少sample_interval频率")

def main():
    """主函数"""
    print("🔍 GPU内存使用诊断工具")
    print("=" * 60)
    
    # 1. 检查系统状态
    check_system_memory()
    check_gpu_memory()
    check_process_memory()
    
    # 2. 测试模型加载
    test_model_loading()
    
    # 3. 分析内存使用
    analyze_memory_usage()
    
    # 4. 建议优化方案
    suggest_optimizations()
    
    print(f"\n✅ 诊断完成")

if __name__ == "__main__":
    main()
