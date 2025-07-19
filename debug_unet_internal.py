#!/usr/bin/env python3
"""
调试UNet内部维度问题
检查UNet内部各层的维度配置
"""

import torch
from pathlib import Path
import sys

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def debug_unet_internal():
    """调试UNet内部配置"""
    print("🔍 调试UNet内部配置...")
    
    # 模型路径
    unet_path = "/kaggle/input/diffusion-final-model"
    
    try:
        from diffusers import UNet2DConditionModel
        
        # 加载UNet
        print(f"从路径加载UNet: {unet_path}")
        unet = UNet2DConditionModel.from_pretrained(unet_path)
        
        print(f"\n📋 UNet完整配置:")
        config = unet.config
        for key, value in config.items():
            print(f"  - {key}: {value}")
        
        print(f"\n🔍 关键维度分析:")
        print(f"  - cross_attention_dim: {config.cross_attention_dim}")
        print(f"  - attention_head_dim: {config.attention_head_dim}")
        print(f"  - block_out_channels: {config.block_out_channels}")
        
        # 检查是否有不一致的配置
        if hasattr(config, 'attention_head_dim'):
            if isinstance(config.attention_head_dim, (list, tuple)):
                print(f"  - attention_head_dim是列表: {config.attention_head_dim}")
            else:
                print(f"  - attention_head_dim是标量: {config.attention_head_dim}")
                
                # 计算每个注意力头的维度
                if config.cross_attention_dim % config.attention_head_dim != 0:
                    print(f"  ⚠️  cross_attention_dim ({config.cross_attention_dim}) 不能被 attention_head_dim ({config.attention_head_dim}) 整除!")
                else:
                    num_heads = config.cross_attention_dim // config.attention_head_dim
                    print(f"  ✅ 注意力头数: {num_heads}")
        
        # 检查是否有其他可能导致1024维度的配置
        suspicious_configs = []
        for key, value in config.items():
            if isinstance(value, (int, list, tuple)):
                if (isinstance(value, int) and value == 1024) or \
                   (isinstance(value, (list, tuple)) and 1024 in value):
                    suspicious_configs.append((key, value))
        
        if suspicious_configs:
            print(f"\n🚨 发现可能导致1024维度的配置:")
            for key, value in suspicious_configs:
                print(f"  - {key}: {value}")
        else:
            print(f"\n✅ 没有发现1024维度的配置")
        
        return config
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        return None

def test_attention_dimensions():
    """测试注意力层的维度计算"""
    print(f"\n🧪 测试注意力层维度计算...")
    
    try:
        # 模拟不同的配置
        configs_to_test = [
            {"cross_attention_dim": 512, "attention_head_dim": 8},
            {"cross_attention_dim": 512, "attention_head_dim": 64},
            {"cross_attention_dim": 1024, "attention_head_dim": 8},
            {"cross_attention_dim": 1024, "attention_head_dim": 64},
        ]
        
        for config in configs_to_test:
            cross_dim = config["cross_attention_dim"]
            head_dim = config["attention_head_dim"]
            
            if cross_dim % head_dim == 0:
                num_heads = cross_dim // head_dim
                print(f"  ✅ cross_dim={cross_dim}, head_dim={head_dim} -> {num_heads} heads")
            else:
                print(f"  ❌ cross_dim={cross_dim}, head_dim={head_dim} -> 不兼容!")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def check_model_state_dict():
    """检查模型状态字典中的维度"""
    print(f"\n🔍 检查模型权重中的维度...")
    
    unet_path = "/kaggle/input/diffusion-final-model"
    
    try:
        from diffusers import UNet2DConditionModel
        
        # 加载UNet
        unet = UNet2DConditionModel.from_pretrained(unet_path)
        
        # 检查状态字典中的权重维度
        suspicious_weights = []
        
        for name, param in unet.named_parameters():
            if any(dim == 1024 for dim in param.shape):
                suspicious_weights.append((name, param.shape))
        
        if suspicious_weights:
            print(f"🚨 发现包含1024维度的权重:")
            for name, shape in suspicious_weights[:10]:  # 只显示前10个
                print(f"  - {name}: {shape}")
            if len(suspicious_weights) > 10:
                print(f"  ... 还有 {len(suspicious_weights) - 10} 个权重")
        else:
            print(f"✅ 没有发现1024维度的权重")
        
        # 特别检查交叉注意力相关的权重
        cross_attn_weights = []
        for name, param in unet.named_parameters():
            if "cross_attn" in name.lower() or "encoder_hidden_states" in name.lower():
                cross_attn_weights.append((name, param.shape))
        
        if cross_attn_weights:
            print(f"\n🔍 交叉注意力相关权重:")
            for name, shape in cross_attn_weights[:5]:  # 只显示前5个
                print(f"  - {name}: {shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def create_minimal_test():
    """创建最小化测试来复现问题"""
    print(f"\n🧪 创建最小化测试...")
    
    try:
        from diffusers import UNet2DConditionModel
        from training.train_diffusion import UserConditionEncoder
        
        # 使用与训练时相同的配置创建UNet
        print("创建UNet (使用训练时配置)...")
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
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
            cross_attention_dim=512,
            attention_head_dim=8,
            use_linear_projection=True,
        )
        
        print("创建条件编码器...")
        condition_encoder = UserConditionEncoder(
            num_users=31,
            embed_dim=512
        )
        
        print("测试前向传播...")
        with torch.no_grad():
            # 创建测试输入
            latents = torch.randn(1, 4, 32, 32)
            timesteps = torch.tensor([100])
            user_tensor = torch.tensor([0])
            
            # 编码条件
            encoder_hidden_states = condition_encoder(user_tensor)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)  # [1, 1, 512]
            
            print(f"输入形状:")
            print(f"  - latents: {latents.shape}")
            print(f"  - timesteps: {timesteps.shape}")
            print(f"  - encoder_hidden_states: {encoder_hidden_states.shape}")
            
            # 测试UNet前向传播
            try:
                noise_pred = unet(
                    latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False
                )[0]
                
                print(f"✅ 测试成功! 输出形状: {noise_pred.shape}")
                return True
                
            except Exception as e:
                print(f"❌ UNet前向传播失败: {e}")
                return False
        
    except Exception as e:
        print(f"❌ 最小化测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 UNet内部维度调试工具")
    print("=" * 50)
    
    # 调试UNet配置
    config = debug_unet_internal()
    
    # 测试注意力维度
    test_attention_dimensions()
    
    # 检查模型权重
    check_model_state_dict()
    
    # 最小化测试
    create_minimal_test()
    
    print("\n" + "=" * 50)
    print("🎯 调试完成!")
    
    if config:
        print("\n💡 建议:")
        print("1. 检查UNet的attention_head_dim配置")
        print("2. 确认cross_attention_dim与attention_head_dim的兼容性")
        print("3. 检查是否有硬编码的1024维度")
        print("4. 考虑重新训练UNet使用正确的配置")

if __name__ == "__main__":
    main()
