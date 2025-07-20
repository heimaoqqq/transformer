#!/usr/bin/env python3
"""
验证模型配置一致性
检查UNet和条件编码器的维度是否匹配
"""

import torch
from pathlib import Path
import sys

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def check_model_consistency():
    """检查模型配置一致性"""
    print("🔍 检查模型配置一致性...")
    
    # 模型路径 - 请根据实际情况修改
    unet_path = "/kaggle/input/diffusion-final-model"
    condition_encoder_path = "/kaggle/input/diffusion-final-model/condition_encoder.pt"
    
    try:
        # 1. 检查UNet配置
        print("\n1️⃣ 检查UNet配置:")
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(unet_path)
        
        print(f"   UNet cross_attention_dim: {unet.config.cross_attention_dim}")
        print(f"   UNet in_channels: {unet.config.in_channels}")
        print(f"   UNet sample_size: {unet.config.sample_size}")
        
        # 2. 检查条件编码器配置
        print("\n2️⃣ 检查条件编码器配置:")
        condition_encoder_state = torch.load(condition_encoder_path, map_location='cpu')
        
        if 'user_embedding.weight' in condition_encoder_state:
            num_users, embed_dim = condition_encoder_state['user_embedding.weight'].shape
            print(f"   条件编码器 num_users: {num_users}")
            print(f"   条件编码器 embed_dim: {embed_dim}")
        else:
            print("   ❌ 无法从权重文件推断条件编码器配置")
            return False
        
        # 3. 检查维度匹配
        print("\n3️⃣ 检查维度匹配:")
        if embed_dim == unet.config.cross_attention_dim:
            print(f"   ✅ 维度匹配! ({embed_dim})")
            return True
        else:
            print(f"   ❌ 维度不匹配!")
            print(f"      条件编码器: {embed_dim}")
            print(f"      UNet期望: {unet.config.cross_attention_dim}")
            
            # 4. 分析可能的原因
            print("\n4️⃣ 可能的原因分析:")
            if unet.config.cross_attention_dim == 1024 and embed_dim == 512:
                print("   🤔 可能原因:")
                print("      - UNet使用了更大的配置 (1024维)")
                print("      - 条件编码器使用了中型配置 (512维)")
                print("      - 这可能是因为UNet和条件编码器来自不同的训练配置")
            
            print("\n   💡 解决方案:")
            print("      1. 使用投影层 (当前修复方案)")
            print("      2. 重新训练条件编码器使用1024维")
            print("      3. 重新训练UNet使用512维")
            
            return False
            
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def simulate_training_scenario():
    """模拟训练时的场景"""
    print("\n🎯 模拟训练时的场景:")
    
    try:
        from ..training.train_diffusion import UserConditionEncoder
        from diffusers import UNet2DConditionModel
        
        # 模拟训练时的配置
        cross_attention_dim = 512  # 假设这是训练时使用的配置
        num_users = 31
        
        print(f"   训练配置: cross_attention_dim = {cross_attention_dim}")
        
        # 创建条件编码器 (训练时的方式)
        condition_encoder = UserConditionEncoder(
            num_users=num_users,
            embed_dim=cross_attention_dim
        )
        
        # 创建UNet (训练时的方式)
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
            cross_attention_dim=cross_attention_dim,  # 关键：使用相同的维度
            attention_head_dim=8,
            use_linear_projection=True,
        )
        
        print(f"   ✅ 训练时配置一致:")
        print(f"      条件编码器 embed_dim: {condition_encoder.embed_dim}")
        print(f"      UNet cross_attention_dim: {unet.config.cross_attention_dim}")
        
        # 测试兼容性
        with torch.no_grad():
            user_tensor = torch.tensor([0])
            encoder_hidden_states = condition_encoder(user_tensor)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            
            latents = torch.randn(1, 4, 32, 32)
            timesteps = torch.tensor([100])
            
            # 这应该不会出错
            noise_pred = unet(
                latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]
            
            print(f"   ✅ 训练时兼容性测试通过!")
            print(f"      输入形状: {encoder_hidden_states.shape}")
            print(f"      输出形状: {noise_pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 训练时场景模拟失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 模型配置一致性验证工具")
    print("=" * 50)
    
    # 检查实际模型配置
    consistency_ok = check_model_consistency()
    
    # 模拟训练时场景
    training_ok = simulate_training_scenario()
    
    print("\n" + "=" * 50)
    print("📊 总结:")
    
    if consistency_ok:
        print("✅ 你的模型配置是一致的，不应该出现维度不匹配问题")
        print("   如果仍然出错，可能是其他原因")
    else:
        print("❌ 你的模型配置不一致，这解释了为什么会出现维度不匹配")
        print("   训练时没问题是因为UNet和条件编码器是同时创建的，配置一致")
        print("   推理时有问题是因为加载的模型来自不同的训练配置")
    
    if training_ok:
        print("✅ 训练时场景模拟成功，证明同时创建时不会有问题")
    
    print("\n💡 建议:")
    print("1. 使用当前的投影层修复方案 (已实现)")
    print("2. 或者确保UNet和条件编码器来自同一次训练")
    print("3. 检查你的训练配置，确认cross_attention_dim设置")

if __name__ == "__main__":
    main()
