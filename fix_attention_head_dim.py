#!/usr/bin/env python3
"""
修复attention_head_dim问题
检查并修复UNet的注意力头维度配置
"""

import torch
from pathlib import Path
import sys

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def analyze_attention_problem():
    """分析注意力维度问题"""
    print("🔍 分析注意力维度问题...")
    
    unet_path = "/kaggle/input/diffusion-final-model"
    
    try:
        from diffusers import UNet2DConditionModel
        
        # 加载UNet
        unet = UNet2DConditionModel.from_pretrained(unet_path)
        config = unet.config
        
        print(f"📋 当前UNet配置:")
        print(f"  - cross_attention_dim: {config.cross_attention_dim}")
        print(f"  - attention_head_dim: {config.attention_head_dim}")
        print(f"  - block_out_channels: {config.block_out_channels}")
        
        # 分析问题
        cross_dim = config.cross_attention_dim
        head_dim = config.attention_head_dim
        
        print(f"\n🔍 维度分析:")
        
        if isinstance(head_dim, (list, tuple)):
            print(f"  attention_head_dim是列表: {head_dim}")
            for i, hd in enumerate(head_dim):
                if cross_dim % hd != 0:
                    print(f"  ❌ 层{i}: cross_attention_dim ({cross_dim}) 不能被 attention_head_dim ({hd}) 整除")
                else:
                    num_heads = cross_dim // hd
                    print(f"  ✅ 层{i}: {num_heads} 个注意力头")
        else:
            print(f"  attention_head_dim是标量: {head_dim}")
            if cross_dim % head_dim != 0:
                print(f"  ❌ cross_attention_dim ({cross_dim}) 不能被 attention_head_dim ({head_dim}) 整除")
                
                # 建议修复方案
                print(f"\n💡 建议的attention_head_dim值:")
                for candidate in [8, 16, 32, 64]:
                    if cross_dim % candidate == 0:
                        num_heads = cross_dim // candidate
                        print(f"    - {candidate} (得到 {num_heads} 个注意力头)")
            else:
                num_heads = cross_dim // head_dim
                print(f"  ✅ {num_heads} 个注意力头")
        
        # 检查block_out_channels的最后一个值
        if hasattr(config, 'block_out_channels'):
            last_channel = config.block_out_channels[-1]
            print(f"\n🔍 检查block_out_channels:")
            print(f"  - 最后一层通道数: {last_channel}")
            
            # 检查是否有1024的通道数
            if 1024 in config.block_out_channels:
                print(f"  ⚠️  发现1024通道数在block_out_channels中!")
                print(f"  这可能是问题的根源")
        
        return config
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None

def suggest_fix():
    """建议修复方案"""
    print(f"\n🔧 修复建议:")
    
    print(f"基于错误信息 'tensor a (512) vs tensor b (1024)'，可能的原因:")
    print(f"1. UNet内部某层期望1024维输入，但收到512维")
    print(f"2. attention_head_dim配置不正确")
    print(f"3. UNet的某个投影层有硬编码的1024维度")
    
    print(f"\n💡 解决方案:")
    print(f"方案1: 修改推理代码，强制使用1024维条件编码器")
    print(f"方案2: 检查UNet训练时的实际配置")
    print(f"方案3: 重新创建UNet使用正确的配置")
    
    print(f"\n🚀 快速修复代码:")
    print(f"```python")
    print(f"# 在inference/generate.py中，强制使用1024维")
    print(f"self.condition_encoder = UserConditionEncoder(")
    print(f"    num_users=num_users,")
    print(f"    embed_dim=1024  # 强制使用1024维")
    print(f")")
    print(f"```")

def create_test_with_1024():
    """测试使用1024维的条件编码器"""
    print(f"\n🧪 测试1024维条件编码器...")
    
    try:
        from training.train_diffusion import UserConditionEncoder
        
        # 创建1024维的条件编码器
        condition_encoder_1024 = UserConditionEncoder(
            num_users=31,
            embed_dim=1024
        )
        
        print(f"✅ 成功创建1024维条件编码器")
        
        # 测试编码
        with torch.no_grad():
            user_tensor = torch.tensor([0])
            encoder_hidden_states = condition_encoder_1024(user_tensor)
            print(f"  输出形状: {encoder_hidden_states.shape}")
            
            if encoder_hidden_states.shape[-1] == 1024:
                print(f"  ✅ 输出维度正确: 1024")
                return True
            else:
                print(f"  ❌ 输出维度错误: {encoder_hidden_states.shape[-1]}")
                return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 注意力头维度修复工具")
    print("=" * 50)
    
    # 分析问题
    config = analyze_attention_problem()
    
    # 建议修复方案
    suggest_fix()
    
    # 测试1024维方案
    test_1024_ok = create_test_with_1024()
    
    print("\n" + "=" * 50)
    print("🎯 总结:")
    
    if config:
        cross_dim = config.cross_attention_dim
        head_dim = config.attention_head_dim
        
        if isinstance(head_dim, int) and cross_dim % head_dim != 0:
            print("❌ 发现attention_head_dim配置问题")
            print(f"   cross_attention_dim ({cross_dim}) 不能被 attention_head_dim ({head_dim}) 整除")
        else:
            print("✅ attention_head_dim配置看起来正常")
    
    if test_1024_ok:
        print("✅ 1024维条件编码器测试成功")
        print("   建议尝试在推理代码中使用1024维")
    else:
        print("❌ 1024维条件编码器测试失败")
    
    print(f"\n🚀 下一步:")
    print(f"1. 尝试修改推理代码使用1024维条件编码器")
    print(f"2. 或者检查UNet训练时的实际配置")
    print(f"3. 运行 debug_unet_internal.py 获取更详细信息")

if __name__ == "__main__":
    main()
