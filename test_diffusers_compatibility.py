#!/usr/bin/env python3
"""
Diffusers库兼容性专项测试
验证所有使用的API和参数是否在当前版本中可用
"""

import warnings
warnings.filterwarnings("ignore")

def test_diffusers_version():
    """测试Diffusers版本"""
    print("🔍 检查Diffusers版本...")
    
    try:
        import diffusers
        version = diffusers.__version__
        print(f"✅ Diffusers版本: {version}")
        
        # 检查是否为推荐版本
        from packaging import version as pkg_version
        if pkg_version.parse(version) >= pkg_version.parse("0.25.0"):
            print("✅ 版本符合要求 (>= 0.25.0)")
            return True
        else:
            print(f"⚠️  版本过低，推荐升级到 >= 0.25.0")
            return False
            
    except ImportError:
        print("❌ Diffusers未安装")
        return False
    except Exception as e:
        print(f"❌ 版本检查失败: {e}")
        return False

def test_autoencoder_kl():
    """测试AutoencoderKL的所有参数"""
    print("\n🔧 测试AutoencoderKL...")
    
    try:
        from diffusers import AutoencoderKL
        
        # 测试我们使用的所有参数
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
        
        print("✅ AutoencoderKL创建成功")
        
        # 测试方法
        import torch
        test_input = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            # 测试encode方法
            posterior = vae.encode(test_input)
            print("✅ encode方法可用")
            
            # 测试latent_dist属性
            latent_dist = posterior.latent_dist
            print("✅ latent_dist属性可用")
            
            # 测试sample方法
            latents = latent_dist.sample()
            print("✅ sample方法可用")
            
            # 测试decode方法
            decoded = vae.decode(latents)
            print("✅ decode方法可用")
            
            # 测试sample属性
            output = decoded.sample
            print("✅ sample属性可用")
            
            # 测试config属性
            scaling_factor = vae.config.scaling_factor
            print(f"✅ config.scaling_factor可用: {scaling_factor}")
        
        # 测试保存和加载
        try:
            vae.save_pretrained("./test_vae")
            loaded_vae = AutoencoderKL.from_pretrained("./test_vae")
            print("✅ save_pretrained/from_pretrained方法可用")
            
            # 清理
            import shutil
            shutil.rmtree("./test_vae")
        except Exception as e:
            print(f"⚠️  保存/加载测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ AutoencoderKL测试失败: {e}")
        return False

def test_unet2d_condition_model():
    """测试UNet2DConditionModel的所有参数"""
    print("\n🔧 测试UNet2DConditionModel...")
    
    try:
        from diffusers import UNet2DConditionModel
        
        # 测试我们使用的所有参数
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(320, 640, 1280, 1280),
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
            cross_attention_dim=768,
            attention_head_dim=8,
            use_linear_projection=True,
            class_embed_type=None,
            num_class_embeds=None,
            upcast_attention=False,
            resnet_time_scale_shift="default",
            time_embedding_type="positional",
            time_embedding_dim=None,
            time_embedding_act_fn=None,
            timestep_post_act=None,
            time_cond_proj_dim=None,
        )
        
        print("✅ UNet2DConditionModel创建成功")
        
        # 测试前向传播
        import torch
        
        with torch.no_grad():
            latents = torch.randn(1, 4, 32, 32)
            timesteps = torch.randint(0, 1000, (1,))
            encoder_hidden_states = torch.randn(1, 1, 768)
            
            # 测试前向传播
            output = unet(
                latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )
            
            print("✅ 前向传播成功")
            print(f"✅ 输出形状: {output[0].shape}")
            
            # 测试return_dict=True
            output_dict = unet(
                latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=True
            )
            
            print("✅ return_dict=True可用")
            print(f"✅ sample属性可用: {output_dict.sample.shape}")
        
        # 测试保存和加载
        try:
            unet.save_pretrained("./test_unet")
            loaded_unet = UNet2DConditionModel.from_pretrained("./test_unet")
            print("✅ save_pretrained/from_pretrained方法可用")
            
            # 清理
            import shutil
            shutil.rmtree("./test_unet")
        except Exception as e:
            print(f"⚠️  保存/加载测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ UNet2DConditionModel测试失败: {e}")
        return False

def test_schedulers():
    """测试调度器"""
    print("\n🔧 测试调度器...")
    
    try:
        from diffusers import DDPMScheduler, DDIMScheduler
        
        # 测试DDPMScheduler
        ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            trained_betas=None,
            variance_type="fixed_small",
            clip_sample=False,
            prediction_type="epsilon",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            clip_sample_range=1.0,
            sample_max_value=1.0,
        )
        
        print("✅ DDPMScheduler创建成功")
        
        # 测试DDIMScheduler
        ddim_scheduler = DDIMScheduler.from_config(ddpm_scheduler.config)
        print("✅ DDIMScheduler.from_config可用")
        
        # 测试调度器方法
        import torch
        
        latents = torch.randn(1, 4, 32, 32)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (1,))
        
        # 测试add_noise
        noisy_latents = ddpm_scheduler.add_noise(latents, noise, timesteps)
        print("✅ add_noise方法可用")
        
        # 测试step
        noise_pred = torch.randn_like(latents)
        step_output = ddpm_scheduler.step(noise_pred, timesteps[0], latents, return_dict=False)
        print("✅ step方法可用")
        
        # 测试set_timesteps
        ddim_scheduler.set_timesteps(50)
        print("✅ set_timesteps方法可用")
        print(f"✅ timesteps属性可用: {len(ddim_scheduler.timesteps)}")
        
        # 测试init_noise_sigma
        init_sigma = ddim_scheduler.init_noise_sigma
        print(f"✅ init_noise_sigma可用: {init_sigma}")
        
        # 测试scale_model_input
        scaled_input = ddim_scheduler.scale_model_input(latents, timesteps[0])
        print("✅ scale_model_input方法可用")
        
        return True
        
    except Exception as e:
        print(f"❌ 调度器测试失败: {e}")
        return False

def test_pipeline_components():
    """测试Pipeline相关组件"""
    print("\n🔧 测试Pipeline组件...")
    
    try:
        # 测试是否可以导入Pipeline基类
        from diffusers import DiffusionPipeline
        print("✅ DiffusionPipeline导入成功")
        
        # 测试StableDiffusionPipeline (用于参考)
        from diffusers import StableDiffusionPipeline
        print("✅ StableDiffusionPipeline导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline组件测试失败: {e}")
        return False

def test_training_utils():
    """测试训练相关工具"""
    print("\n🔧 测试训练工具...")
    
    try:
        # 测试EMA (如果使用)
        try:
            from diffusers import EMAModel
            print("✅ EMAModel可用")
        except ImportError:
            print("⚠️  EMAModel不可用 (可选)")
        
        # 测试优化相关
        from diffusers.optimization import get_scheduler
        print("✅ get_scheduler可用")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练工具测试失败: {e}")
        return False

def test_full_workflow():
    """测试完整工作流程"""
    print("\n🔧 测试完整工作流程...")
    
    try:
        import torch
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            sample_size=256,
        ).to(device)
        
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            cross_attention_dim=768,
        ).to(device)
        
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(50)
        
        print("✅ 所有模型创建成功")
        
        # 模拟完整的生成流程
        with torch.no_grad():
            # 1. 编码图像
            test_image = torch.randn(1, 3, 256, 256).to(device)
            latents = vae.encode(test_image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            print("✅ 图像编码成功")
            
            # 2. 添加噪声 (训练时)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (1,)).to(device)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            
            print("✅ 噪声添加成功")
            
            # 3. 条件编码
            encoder_hidden_states = torch.randn(1, 1, 768).to(device)
            
            # 4. UNet预测
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            print("✅ UNet预测成功")
            
            # 5. 去噪过程 (推理时)
            latents_gen = torch.randn(1, 4, 32, 32).to(device)
            latents_gen = latents_gen * scheduler.init_noise_sigma
            
            for t in scheduler.timesteps[:5]:  # 只测试前5步
                latent_model_input = scheduler.scale_model_input(latents_gen, t)
                
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                
                latents_gen = scheduler.step(noise_pred, t, latents_gen).prev_sample
            
            print("✅ 去噪过程成功")
            
            # 6. 解码图像
            latents_gen = latents_gen / vae.config.scaling_factor
            generated_image = vae.decode(latents_gen).sample
            
            print("✅ 图像解码成功")
            print(f"✅ 完整流程: {test_image.shape} -> {latents.shape} -> {generated_image.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 完整工作流程测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 Diffusers库兼容性专项测试")
    print("=" * 50)
    
    tests = [
        ("Diffusers版本", test_diffusers_version),
        ("AutoencoderKL", test_autoencoder_kl),
        ("UNet2DConditionModel", test_unet2d_condition_model),
        ("调度器", test_schedulers),
        ("Pipeline组件", test_pipeline_components),
        ("训练工具", test_training_utils),
        ("完整工作流程", test_full_workflow),
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
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 项测试通过")
    
    if passed == len(results):
        print("\n🎉 所有Diffusers API测试通过！")
        print("✅ 可以安全地使用当前版本进行训练")
    else:
        print("\n⚠️  部分测试失败")
        print("建议:")
        print("1. 升级Diffusers: pip install --upgrade diffusers>=0.25.0")
        print("2. 检查PyTorch版本兼容性")
        print("3. 重新运行测试")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
