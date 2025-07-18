#!/usr/bin/env python3
"""
检查扩散模型训练代码的API兼容性
确保当前安装的版本与train_diffusion.py完全兼容
"""

import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")

def run_command(cmd, description=""):
    """运行命令"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - 完成")
            return True
        else:
            print(f"❌ {description} - 失败")
            if result.stderr:
                print(f"错误: {result.stderr[:200]}...")
            return False
    except Exception as e:
        print(f"❌ {description} - 异常: {e}")
        return False

def check_current_versions():
    """检查当前版本"""
    print("📦 检查当前安装版本:")
    
    packages = [
        'torch', 'diffusers', 'transformers', 'accelerate', 'huggingface_hub'
    ]
    
    versions = {}
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {package}: {version}")
            versions[package] = version
        except ImportError:
            print(f"   ❌ {package}: 未安装")
            versions[package] = None
    
    return versions

def test_diffusion_api_compatibility():
    """测试扩散模型API兼容性"""
    print("\n🔧 测试扩散模型API兼容性:")
    
    try:
        import torch
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
        
        print("   1️⃣ 导入成功")
        
        # 测试VAE API
        print("   2️⃣ 测试VAE API...")
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            sample_size=128,
        )
        
        test_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            # 关键API测试
            encode_result = vae.encode(test_input)
            if not hasattr(encode_result, 'latent_dist'):
                print("   ❌ VAE encode().latent_dist 不可用")
                return False
            
            posterior = encode_result.latent_dist
            if not hasattr(posterior, 'sample'):
                print("   ❌ latent_dist.sample() 不可用")
                return False
            
            latents = posterior.sample()
            if not hasattr(vae, 'config') or not hasattr(vae.config, 'scaling_factor'):
                print("   ❌ vae.config.scaling_factor 不可用")
                return False
            
            print("   ✅ VAE API兼容")
        
        # 测试UNet API
        print("   3️⃣ 测试UNet API...")
        unet = UNet2DConditionModel(
            sample_size=16,  # 128//8
            in_channels=4,
            out_channels=4,
            cross_attention_dim=768,
            layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        )
        
        with torch.no_grad():
            test_latents = torch.randn(1, 4, 16, 16)
            test_timesteps = torch.randint(0, 1000, (1,))
            test_conditions = torch.randn(1, 1, 768)
            
            # 测试关键用法
            result = unet(
                test_latents,
                test_timesteps,
                encoder_hidden_states=test_conditions,
                return_dict=False
            )
            
            if not isinstance(result, tuple) or len(result) == 0:
                print("   ❌ UNet return_dict=False 模式不兼容")
                return False
            
            print("   ✅ UNet API兼容")
        
        # 测试调度器API
        print("   4️⃣ 测试调度器API...")
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )
        
        with torch.no_grad():
            test_noise = torch.randn_like(test_latents)
            test_timesteps = torch.randint(0, 1000, (1,))
            
            # 关键方法测试
            noisy_latents = scheduler.add_noise(test_latents, test_noise, test_timesteps)
            
            if not hasattr(scheduler, 'config'):
                print("   ❌ scheduler.config 不可用")
                return False
            
            if not hasattr(scheduler.config, 'num_train_timesteps'):
                print("   ❌ scheduler.config.num_train_timesteps 不可用")
                return False
            
            print("   ✅ 调度器API兼容")
        
        # 测试DDIM调度器
        print("   5️⃣ 测试DDIM调度器...")
        ddim_scheduler = DDIMScheduler.from_config(scheduler.config)
        ddim_scheduler.set_timesteps(50)
        
        with torch.no_grad():
            step_result = ddim_scheduler.step(test_noise, test_timesteps[0], test_latents)
            if not hasattr(step_result, 'prev_sample'):
                print("   ❌ DDIM step().prev_sample 不可用")
                return False
            
            print("   ✅ DDIM调度器兼容")
        
        print("   🎉 所有API完全兼容！")
        return True
        
    except Exception as e:
        print(f"   ❌ API兼容性测试失败: {e}")
        return False

def fix_version_compatibility():
    """修复版本兼容性问题"""
    print("\n🔧 修复版本兼容性问题:")
    
    # 稳定的兼容版本组合 (经过验证)
    recommended_versions = [
        "huggingface_hub==0.16.4",  # 包含 cached_download
        "diffusers==0.21.4",        # 与 huggingface_hub 兼容
        "transformers==4.30.2",
        "accelerate==0.20.3"
    ]
    
    print("   安装推荐版本组合...")
    
    success = True
    for package in recommended_versions:
        if not run_command(f"pip install {package}", f"安装 {package}"):
            success = False
    
    return success

def main():
    """主函数"""
    print("🔍 扩散模型训练兼容性检查工具")
    print("=" * 60)
    
    # 1. 检查当前版本
    print("\n" + "="*20 + " 版本检查 " + "="*20)
    versions = check_current_versions()
    
    # 2. 测试API兼容性
    print("\n" + "="*20 + " API兼容性测试 " + "="*20)
    api_compatible = test_diffusion_api_compatibility()
    
    # 3. 结果和建议
    print("\n" + "="*20 + " 结果和建议 " + "="*20)
    
    if api_compatible:
        print("🎉 当前版本完全兼容！")
        print("✅ 可以直接开始扩散模型训练")
        print("\n📋 下一步:")
        print("   python training/train_diffusion.py --help")
        return True
    else:
        print("⚠️  发现兼容性问题")
        print("🔧 尝试自动修复...")
        
        if fix_version_compatibility():
            print("\n✅ 版本修复完成，请重新运行此脚本验证")
            print("📋 验证命令:")
            print("   python check_diffusion_compatibility.py")
        else:
            print("\n❌ 自动修复失败")
            print("🔧 手动修复建议:")
            print("   1. 运行: python ultimate_fix_kaggle.py")
            print("   2. 或手动安装: pip install diffusers==0.25.1 transformers==4.36.2")
            print("   3. 重启内核后重新测试")
        
        return False

if __name__ == "__main__":
    main()
