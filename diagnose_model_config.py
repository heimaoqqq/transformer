#!/usr/bin/env python3
"""
诊断模型配置脚本
检查UNet和条件编码器的维度匹配问题
"""

import torch
from pathlib import Path
import sys

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def diagnose_unet_config(unet_path):
    """诊断UNet配置"""
    print("🔍 诊断UNet配置...")
    
    try:
        from diffusers import UNet2DConditionModel
        
        # 加载UNet
        print(f"从路径加载UNet: {unet_path}")
        unet = UNet2DConditionModel.from_pretrained(unet_path)
        
        print(f"\n📋 UNet配置信息:")
        print(f"  - cross_attention_dim: {unet.config.cross_attention_dim}")
        print(f"  - in_channels: {unet.config.in_channels}")
        print(f"  - out_channels: {unet.config.out_channels}")
        print(f"  - sample_size: {unet.config.sample_size}")
        print(f"  - layers_per_block: {unet.config.layers_per_block}")
        print(f"  - block_out_channels: {unet.config.block_out_channels}")
        print(f"  - attention_head_dim: {unet.config.attention_head_dim}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in unet.parameters())
        print(f"  - 总参数量: {total_params:,}")
        
        return unet.config.cross_attention_dim
        
    except Exception as e:
        print(f"❌ UNet配置诊断失败: {e}")
        return None

def diagnose_condition_encoder(condition_encoder_path, expected_embed_dim, num_users=31):
    """诊断条件编码器配置"""
    print(f"\n🎭 诊断条件编码器配置...")
    
    try:
        from training.train_diffusion import UserConditionEncoder
        
        # 尝试加载条件编码器
        print(f"从路径加载条件编码器: {condition_encoder_path}")
        
        # 处理路径
        if Path(condition_encoder_path).is_dir():
            condition_encoder_file = Path(condition_encoder_path) / "condition_encoder.pt"
            if not condition_encoder_file.exists():
                print(f"❌ 条件编码器文件不存在: {condition_encoder_file}")
                return False
            condition_encoder_path = str(condition_encoder_file)
        
        # 创建条件编码器实例
        condition_encoder = UserConditionEncoder(
            num_users=num_users,
            embed_dim=expected_embed_dim
        )
        
        # 加载权重
        state_dict = torch.load(condition_encoder_path, map_location='cpu')
        
        print(f"\n📋 条件编码器状态字典信息:")
        for key, tensor in state_dict.items():
            print(f"  - {key}: {tensor.shape}")
        
        # 检查嵌入层维度
        if 'user_embedding.weight' in state_dict:
            embed_shape = state_dict['user_embedding.weight'].shape
            actual_num_users, actual_embed_dim = embed_shape
            
            print(f"\n📏 嵌入层维度分析:")
            print(f"  - 实际用户数: {actual_num_users}")
            print(f"  - 实际嵌入维度: {actual_embed_dim}")
            print(f"  - 期望用户数: {num_users}")
            print(f"  - 期望嵌入维度: {expected_embed_dim}")
            
            if actual_embed_dim != expected_embed_dim:
                print(f"⚠️  维度不匹配!")
                print(f"   条件编码器嵌入维度: {actual_embed_dim}")
                print(f"   UNet期望维度: {expected_embed_dim}")
                return False
            else:
                print(f"✅ 维度匹配!")
                return True
        else:
            print(f"❌ 找不到user_embedding.weight")
            return False
            
    except Exception as e:
        print(f"❌ 条件编码器诊断失败: {e}")
        return False

def suggest_fix(unet_cross_attention_dim, condition_encoder_embed_dim):
    """建议修复方案"""
    print(f"\n🔧 修复建议:")
    
    if unet_cross_attention_dim != condition_encoder_embed_dim:
        print(f"❌ 维度不匹配问题:")
        print(f"   UNet cross_attention_dim: {unet_cross_attention_dim}")
        print(f"   条件编码器 embed_dim: {condition_encoder_embed_dim}")
        
        print(f"\n💡 解决方案:")
        print(f"1. 重新训练条件编码器，使用embed_dim={unet_cross_attention_dim}")
        print(f"2. 或者重新训练UNet，使用cross_attention_dim={condition_encoder_embed_dim}")
        print(f"3. 或者在推理时创建新的条件编码器，使用正确的维度")
        
        print(f"\n🚀 快速修复 (推荐):")
        print(f"在inference/generate.py中，修改条件编码器创建:")
        print(f"```python")
        print(f"# 方案1: 使用UNet的cross_attention_dim")
        print(f"self.condition_encoder = UserConditionEncoder(")
        print(f"    num_users=num_users,")
        print(f"    embed_dim={unet_cross_attention_dim}  # 使用UNet的维度")
        print(f")")
        print(f"```")
        
        print(f"\n⚠️  注意: 这需要重新训练条件编码器或使用兼容的预训练权重")
    else:
        print(f"✅ 维度匹配，问题可能在其他地方")

def main():
    """主诊断函数"""
    print("🔍 模型配置诊断工具")
    print("=" * 50)
    
    # 示例路径 - 请根据实际情况修改
    unet_path = "/kaggle/input/diffusion-final-model"
    condition_encoder_path = "/kaggle/input/diffusion-final-model/condition_encoder.pt"
    num_users = 31
    
    print(f"📁 诊断路径:")
    print(f"  UNet路径: {unet_path}")
    print(f"  条件编码器路径: {condition_encoder_path}")
    print(f"  用户数量: {num_users}")
    
    # 诊断UNet
    unet_cross_attention_dim = diagnose_unet_config(unet_path)
    
    if unet_cross_attention_dim is None:
        print("❌ 无法获取UNet配置，停止诊断")
        return
    
    # 诊断条件编码器
    condition_encoder_ok = diagnose_condition_encoder(
        condition_encoder_path, 
        unet_cross_attention_dim, 
        num_users
    )
    
    # 提供修复建议
    suggest_fix(unet_cross_attention_dim, unet_cross_attention_dim if condition_encoder_ok else "未知")
    
    print(f"\n" + "=" * 50)
    print(f"🎯 诊断完成!")

if __name__ == "__main__":
    main()
