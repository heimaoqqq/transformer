#!/usr/bin/env python3
"""
LDM API确认脚本
验证扩散模型相关API的兼容性和可用性
"""

import torch
import warnings
warnings.filterwarnings("ignore")

def check_versions():
    """检查关键包版本"""
    print("📦 检查关键包版本:")
    
    packages = [
        ('torch', 'PyTorch'),
        ('diffusers', 'Diffusers'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
    ]
    
    versions = {}
    for package, name in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {name}: {version}")
            versions[package] = version
        except ImportError:
            print(f"   ❌ {name}: 未安装")
            versions[package] = None
    
    return versions

def test_unet_api():
    """测试UNet2DConditionModel API"""
    print("\n🎯 测试UNet2DConditionModel API:")
    
    try:
        from diffusers import UNet2DConditionModel
        
        # 1. 测试构造函数
        print("   1️⃣ 测试构造函数...")
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            cross_attention_dim=512,  # 中型配置
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
            attention_head_dim=8,
            use_linear_projection=True,
        )
        print("   ✅ UNet构造函数兼容")
        
        # 2. 测试前向传播
        print("   2️⃣ 测试前向传播...")
        with torch.no_grad():
            sample = torch.randn(1, 4, 32, 32)
            timestep = torch.randint(0, 1000, (1,))
            encoder_hidden_states = torch.randn(1, 1, 512)  # 匹配新的cross_attention_dim
            
            # 测试return_dict=False模式
            result = unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )
            
            if isinstance(result, tuple) and len(result) > 0:
                noise_pred = result[0]
                print("   ✅ return_dict=False 模式兼容")
            else:
                print("   ❌ return_dict=False 模式不兼容")
                return False
            
            # 测试.sample属性模式
            result2 = unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states
            )
            
            if hasattr(result2, 'sample'):
                noise_pred2 = result2.sample
                print("   ✅ .sample 属性兼容")
            else:
                print("   ❌ .sample 属性不兼容")
                return False
        
        # 3. 测试配置属性
        print("   3️⃣ 测试配置属性...")
        config_attrs = [
            'sample_size', 'in_channels', 'out_channels', 
            'cross_attention_dim', 'layers_per_block'
        ]
        
        for attr in config_attrs:
            if hasattr(unet.config, attr):
                value = getattr(unet.config, attr)
                print(f"      ✅ config.{attr}: {value}")
            else:
                print(f"      ❌ config.{attr}: 不可用")
                return False
        
        print("   ✅ UNet API完全兼容")
        return True
        
    except Exception as e:
        print(f"   ❌ UNet API测试失败: {e}")
        return False

def test_scheduler_api():
    """测试调度器API"""
    print("\n⏰ 测试调度器API:")
    
    try:
        from diffusers import DDPMScheduler, DDIMScheduler
        
        # 1. 测试DDPM调度器
        print("   1️⃣ 测试DDPM调度器...")
        ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            variance_type="fixed_small",
            clip_sample=False,
            prediction_type="epsilon",
        )
        
        # 测试关键方法
        test_latents = torch.randn(1, 4, 32, 32)
        test_noise = torch.randn_like(test_latents)
        test_timesteps = torch.randint(0, 1000, (1,))
        
        with torch.no_grad():
            # add_noise方法
            noisy_latents = ddpm_scheduler.add_noise(test_latents, test_noise, test_timesteps)
            print("      ✅ add_noise() 方法可用")
            
            # scale_model_input方法
            scaled_input = ddpm_scheduler.scale_model_input(test_latents, test_timesteps[0])
            print("      ✅ scale_model_input() 方法可用")
            
            # step方法
            step_result = ddpm_scheduler.step(test_noise, test_timesteps[0], test_latents)
            if hasattr(step_result, 'prev_sample'):
                print("      ✅ step().prev_sample 可用")
            else:
                print("      ❌ step().prev_sample 不可用")
                return False
            
            # 配置属性
            config_attrs = ['num_train_timesteps', 'beta_start', 'beta_end', 'prediction_type']
            for attr in config_attrs:
                if hasattr(ddpm_scheduler.config, attr):
                    value = getattr(ddpm_scheduler.config, attr)
                    print(f"      ✅ config.{attr}: {value}")
                else:
                    print(f"      ❌ config.{attr}: 不可用")
        
        # 2. 测试DDIM调度器
        print("   2️⃣ 测试DDIM调度器...")
        ddim_scheduler = DDIMScheduler.from_config(ddpm_scheduler.config)
        ddim_scheduler.set_timesteps(50)
        
        with torch.no_grad():
            step_result = ddim_scheduler.step(test_noise, test_timesteps[0], test_latents)
            if hasattr(step_result, 'prev_sample'):
                print("      ✅ DDIM step().prev_sample 可用")
            else:
                print("      ❌ DDIM step().prev_sample 不可用")
                return False
        
        print("   ✅ 调度器API完全兼容")
        return True
        
    except Exception as e:
        print(f"   ❌ 调度器API测试失败: {e}")
        return False

def test_accelerate_api():
    """测试Accelerate API"""
    print("\n🚀 测试Accelerate API:")
    
    try:
        from accelerate import Accelerator
        
        # 创建加速器
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="fp16",
        )
        
        print("   ✅ Accelerator创建成功")
        print(f"      设备: {accelerator.device}")
        print(f"      混合精度: {accelerator.mixed_precision}")
        print(f"      是否主进程: {accelerator.is_main_process}")
        
        # 测试prepare方法
        dummy_model = torch.nn.Linear(10, 1)
        dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
        
        model, optimizer = accelerator.prepare(dummy_model, dummy_optimizer)
        print("   ✅ prepare() 方法可用")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Accelerate API测试失败: {e}")
        return False

def test_training_workflow():
    """测试完整训练工作流程API"""
    print("\n🔄 测试完整训练工作流程API:")
    
    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        from accelerate import Accelerator
        
        # 创建模型
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            sample_size=128,
        )
        
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            cross_attention_dim=512,  # 中型配置
        )
        
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        accelerator = Accelerator()
        
        print("   1️⃣ 模型创建成功")
        
        # 模拟训练步骤
        with torch.no_grad():
            # 输入数据
            images = torch.randn(2, 3, 128, 128)
            user_conditions = torch.randn(2, 1, 512)  # 匹配新的cross_attention_dim
            
            # VAE编码
            posterior = vae.encode(images).latent_dist
            latents = posterior.sample()
            latents = latents * vae.config.scaling_factor
            print("   2️⃣ VAE编码成功")
            
            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],))
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            print("   3️⃣ 噪声添加成功")
            
            # UNet预测
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=user_conditions,
                return_dict=False
            )[0]
            print("   4️⃣ UNet预测成功")
            
            # 损失计算
            loss = torch.nn.functional.mse_loss(model_pred, noise)
            print(f"   5️⃣ 损失计算成功: {loss.item():.6f}")
            
            # VAE解码
            latents_decoded = latents / vae.config.scaling_factor
            reconstruction = vae.decode(latents_decoded).sample
            print("   6️⃣ VAE解码成功")
        
        print("   ✅ 完整训练工作流程API兼容")
        return True
        
    except Exception as e:
        print(f"   ❌ 训练工作流程API测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 LDM API兼容性验证")
    print("=" * 60)
    
    # 1. 检查版本
    versions = check_versions()
    
    # 2. API兼容性测试
    tests = [
        ("UNet2DConditionModel API", test_unet_api),
        ("调度器 API", test_scheduler_api),
        ("Accelerate API", test_accelerate_api),
        ("完整训练工作流程 API", test_training_workflow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 3. 总结
    print("\n" + "="*60)
    print("📊 LDM API兼容性验证总结:")
    
    passed = 0
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\n总体结果: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有LDM API完全兼容！")
        print("✅ 当前版本可以安全用于LDM训练")
        return True
    else:
        print("\n⚠️  存在LDM API兼容性问题")
        print("❌ 建议检查版本或使用推荐版本组合")
        return False

if __name__ == "__main__":
    main()
