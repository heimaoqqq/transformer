#!/usr/bin/env python3
"""
验证当前安装的版本与训练代码的API兼容性
确保所有参数、方法和返回值都正确
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
        ('huggingface_hub', 'HuggingFace Hub')
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

def test_autoencoder_kl_api():
    """测试AutoencoderKL API兼容性"""
    print("\n🔧 测试AutoencoderKL API兼容性:")
    
    try:
        from diffusers import AutoencoderKL
        
        # 1. 测试构造函数参数 (训练代码中使用的)
        print("   1️⃣ 测试构造函数参数...")
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D", 
                "DownEncoderBlock2D",
                "DownEncoderBlock2D"
            ],
            up_block_types=[
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"
            ],
            block_out_channels=[128, 256, 512, 512],
            latent_channels=4,
            sample_size=256,
            layers_per_block=2,
            act_fn="silu",
            norm_num_groups=32,
            scaling_factor=0.18215,
        )
        print("   ✅ 构造函数参数兼容")
        
        # 2. 测试encode方法
        print("   2️⃣ 测试encode方法...")
        test_input = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            # 训练代码中的用法: vae.encode(images).latent_dist
            encode_result = vae.encode(test_input)
            
            # 检查返回值结构
            if hasattr(encode_result, 'latent_dist'):
                posterior = encode_result.latent_dist
                print("   ✅ encode().latent_dist 可用")
                
                # 检查posterior方法
                if hasattr(posterior, 'sample'):
                    latents = posterior.sample()
                    print("   ✅ latent_dist.sample() 可用")
                else:
                    print("   ❌ latent_dist.sample() 不可用")
                    return False
                
                if hasattr(posterior, 'kl'):
                    kl_loss = posterior.kl()
                    print("   ✅ latent_dist.kl() 可用")
                else:
                    print("   ❌ latent_dist.kl() 不可用")
                    return False
                    
            else:
                print("   ❌ encode().latent_dist 不可用")
                return False
        
        # 3. 测试decode方法
        print("   3️⃣ 测试decode方法...")
        with torch.no_grad():
            # 训练代码中的用法: vae.decode(latents).sample
            decode_result = vae.decode(latents)
            
            if hasattr(decode_result, 'sample'):
                reconstruction = decode_result.sample
                print("   ✅ decode().sample 可用")
            else:
                print("   ❌ decode().sample 不可用")
                return False
        
        # 4. 测试config属性
        print("   4️⃣ 测试config属性...")
        if hasattr(vae, 'config'):
            config = vae.config
            if hasattr(config, 'scaling_factor'):
                scaling_factor = config.scaling_factor
                print(f"   ✅ config.scaling_factor 可用: {scaling_factor}")
            else:
                print("   ❌ config.scaling_factor 不可用")
                return False
        else:
            print("   ❌ config 属性不可用")
            return False
        
        print("   ✅ AutoencoderKL API完全兼容")
        return True
        
    except Exception as e:
        print(f"   ❌ AutoencoderKL API测试失败: {e}")
        return False

def test_unet_api():
    """测试UNet2DConditionModel API兼容性"""
    print("\n🎯 测试UNet2DConditionModel API兼容性:")
    
    try:
        from diffusers import UNet2DConditionModel
        
        # 1. 测试构造函数参数
        print("   1️⃣ 测试构造函数参数...")
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            cross_attention_dim=768,
            layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        )
        print("   ✅ 构造函数参数兼容")
        
        # 2. 测试前向传播
        print("   2️⃣ 测试前向传播...")
        with torch.no_grad():
            test_latents = torch.randn(1, 4, 32, 32)
            test_timesteps = torch.randint(0, 1000, (1,))
            test_conditions = torch.randn(1, 1, 768)
            
            # 训练代码中的用法1: return_dict=False
            result1 = unet(
                test_latents,
                test_timesteps,
                encoder_hidden_states=test_conditions,
                return_dict=False
            )
            
            if isinstance(result1, tuple) and len(result1) > 0:
                noise_pred1 = result1[0]
                print("   ✅ return_dict=False 模式兼容")
            else:
                print("   ❌ return_dict=False 模式不兼容")
                return False
            
            # 训练代码中的用法2: .sample属性
            result2 = unet(
                test_latents,
                test_timesteps,
                encoder_hidden_states=test_conditions
            )
            
            if hasattr(result2, 'sample'):
                noise_pred2 = result2.sample
                print("   ✅ .sample 属性兼容")
            else:
                print("   ❌ .sample 属性不兼容")
                return False
        
        # 3. 测试config属性
        print("   3️⃣ 测试config属性...")
        if hasattr(unet, 'config'):
            config = unet.config
            if hasattr(config, 'in_channels'):
                print(f"   ✅ config.in_channels 可用: {config.in_channels}")
            else:
                print("   ❌ config.in_channels 不可用")
                return False
        else:
            print("   ❌ config 属性不可用")
            return False
        
        print("   ✅ UNet2DConditionModel API完全兼容")
        return True
        
    except Exception as e:
        print(f"   ❌ UNet2DConditionModel API测试失败: {e}")
        return False

def test_scheduler_api():
    """测试调度器API兼容性"""
    print("\n⏰ 测试调度器API兼容性:")
    
    try:
        from diffusers import DDPMScheduler, DDIMScheduler
        
        # 1. 测试DDPMScheduler
        print("   1️⃣ 测试DDPMScheduler...")
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )
        
        # 测试关键方法
        test_latents = torch.randn(1, 4, 32, 32)
        test_noise = torch.randn_like(test_latents)
        test_timesteps = torch.randint(0, 1000, (1,))
        
        with torch.no_grad():
            # add_noise方法
            noisy_latents = scheduler.add_noise(test_latents, test_noise, test_timesteps)
            print("   ✅ add_noise() 方法可用")
            
            # config属性
            if hasattr(scheduler, 'config'):
                config = scheduler.config
                if hasattr(config, 'num_train_timesteps'):
                    print(f"   ✅ config.num_train_timesteps 可用: {config.num_train_timesteps}")
                if hasattr(config, 'prediction_type'):
                    print(f"   ✅ config.prediction_type 可用: {config.prediction_type}")
                else:
                    print("   ⚠️  config.prediction_type 不可用 (可能使用默认值)")
            
            # init_noise_sigma属性
            if hasattr(scheduler, 'init_noise_sigma'):
                print(f"   ✅ init_noise_sigma 可用: {scheduler.init_noise_sigma}")
            else:
                print("   ❌ init_noise_sigma 不可用")
                return False
            
            # timesteps属性
            if hasattr(scheduler, 'timesteps'):
                print(f"   ✅ timesteps 可用: 长度 {len(scheduler.timesteps)}")
            else:
                print("   ❌ timesteps 不可用")
                return False
            
            # scale_model_input方法
            scaled_input = scheduler.scale_model_input(test_latents, test_timesteps[0])
            print("   ✅ scale_model_input() 方法可用")
            
            # step方法
            step_result = scheduler.step(test_noise, test_timesteps[0], test_latents)
            if hasattr(step_result, 'prev_sample'):
                print("   ✅ step().prev_sample 可用")
            else:
                print("   ❌ step().prev_sample 不可用")
                return False
        
        # 2. 测试DDIMScheduler
        print("   2️⃣ 测试DDIMScheduler...")
        ddim_scheduler = DDIMScheduler.from_config(scheduler.config)
        ddim_scheduler.set_timesteps(50)
        print("   ✅ DDIMScheduler.from_config() 和 set_timesteps() 可用")
        
        print("   ✅ 调度器API完全兼容")
        return True
        
    except Exception as e:
        print(f"   ❌ 调度器API测试失败: {e}")
        return False

def test_training_workflow():
    """测试完整训练工作流程"""
    print("\n🔄 测试完整训练工作流程:")
    
    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        
        # 创建模型 (小尺寸)
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            sample_size=64,
        )
        
        unet = UNet2DConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            cross_attention_dim=768,
        )
        
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        print("   1️⃣ 模型创建成功")
        
        # 模拟训练步骤
        with torch.no_grad():
            # 输入数据
            images = torch.randn(2, 3, 64, 64)
            user_indices = torch.tensor([0, 1])
            
            # VAE编码 (训练代码中的用法)
            posterior = vae.encode(images).latent_dist
            latents = posterior.sample()
            latents = latents * vae.config.scaling_factor
            
            print("   2️⃣ VAE编码成功")
            
            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],))
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            
            print("   3️⃣ 噪声添加成功")
            
            # 条件编码 (模拟)
            encoder_hidden_states = torch.randn(2, 1, 768)
            
            # UNet预测
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]
            
            print("   4️⃣ UNet预测成功")
            
            # VAE解码
            latents_decoded = latents / vae.config.scaling_factor
            reconstruction = vae.decode(latents_decoded).sample
            
            print("   5️⃣ VAE解码成功")
            
            # 验证形状
            print(f"   📊 形状验证:")
            print(f"      输入图像: {images.shape}")
            print(f"      潜在表示: {latents.shape}")
            print(f"      噪声预测: {model_pred.shape}")
            print(f"      重建图像: {reconstruction.shape}")
        
        print("   ✅ 完整训练工作流程兼容")
        return True
        
    except Exception as e:
        print(f"   ❌ 训练工作流程测试失败: {e}")
        return False

def main():
    """主验证函数"""
    print("🔍 API兼容性验证工具")
    print("验证当前版本与训练代码的完全兼容性")
    print("=" * 60)
    
    # 1. 检查版本
    versions = check_versions()
    
    # 2. API兼容性测试
    tests = [
        ("AutoencoderKL API", test_autoencoder_kl_api),
        ("UNet2DConditionModel API", test_unet_api),
        ("调度器 API", test_scheduler_api),
        ("完整训练工作流程", test_training_workflow)
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
    print("📊 兼容性验证总结:")
    
    passed = 0
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\n总体结果: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有API完全兼容！")
        print("✅ 当前版本可以安全用于训练")
        print("\n📋 版本信息:")
        for package, version in versions.items():
            if version:
                print(f"   - {package}: {version}")
        return True
    else:
        print("\n⚠️  存在API兼容性问题")
        print("❌ 建议使用推荐版本组合")
        return False

if __name__ == "__main__":
    main()
