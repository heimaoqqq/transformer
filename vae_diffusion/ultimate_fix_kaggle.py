#!/usr/bin/env python3
"""
终极Kaggle环境修复脚本
彻底清理并重建所有依赖，解决所有版本冲突
"""

import subprocess
import sys
import os
import shutil

def run_command(cmd, description="", ignore_errors=False):
    """运行命令"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 or ignore_errors:
            print(f"✅ {description} - 完成")
            return True
        else:
            print(f"❌ {description} - 失败")
            if result.stderr and not ignore_errors:
                print(f"错误: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - 异常: {e}")
        return False

def nuclear_cleanup():
    """核弹级清理 - 彻底清除所有相关包"""
    print("💥 核弹级清理 - 彻底清除所有相关包")
    print("=" * 50)
    
    # 1. 清理Python缓存
    print("\n1️⃣ 清理Python缓存...")
    try:
        import sys
        for module in list(sys.modules.keys()):
            if any(pkg in module for pkg in ['numpy', 'torch', 'diffusers', 'transformers', 'scipy', 'sklearn']):
                del sys.modules[module]
        print("✅ Python模块缓存已清理")
    except:
        pass
    
    # 2. 卸载所有相关包
    print("\n2️⃣ 卸载所有相关包...")
    packages_to_remove = [
        # 核心包
        "torch", "torchvision", "torchaudio", "torchtext",
        "numpy", "scipy", "scikit-learn", "matplotlib",
        # AI包
        "diffusers", "transformers", "accelerate", "huggingface_hub",
        # 依赖包
        "pillow", "opencv-python", "opencv-contrib-python",
        "einops", "tqdm", "packaging", "filelock",
        # 可能冲突的包
        "jax", "jaxlib", "flax", "optax",
        "tensorflow", "tensorflow-gpu",
        "pandas", "seaborn"
    ]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall -y {package}", f"卸载 {package}", ignore_errors=True)
    
    # 3. 强制清理pip缓存
    print("\n3️⃣ 清理pip缓存...")
    run_command("pip cache purge", "清理pip缓存", ignore_errors=True)
    
    # 4. 清理conda缓存 (如果存在)
    print("\n4️⃣ 清理conda缓存...")
    run_command("conda clean -a -y", "清理conda缓存", ignore_errors=True)

def install_base_system():
    """安装基础系统包"""
    print("\n🏗️  安装基础系统包")
    print("=" * 30)
    
    # 1. 升级pip和setuptools
    print("\n1️⃣ 升级基础工具...")
    run_command("pip install --upgrade pip setuptools wheel", "升级pip和setuptools")
    
    # 2. 安装NumPy (最稳定版本)
    print("\n2️⃣ 安装NumPy...")
    numpy_versions = ["1.24.4", "1.23.5", "1.21.6"]
    for version in numpy_versions:
        if run_command(f"pip install numpy=={version}", f"安装NumPy {version}"):
            break
    
    # 3. 安装SciPy (兼容NumPy)
    print("\n3️⃣ 安装SciPy...")
    run_command("pip install scipy==1.10.1", "安装SciPy")

def check_gpu_environment():
    """检查GPU环境"""
    print("\n🔍 检查GPU环境")
    print("=" * 30)

    # 1. 检查nvidia-smi
    print("\n1️⃣ 检查nvidia-smi...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi可用")
            # 提取GPU信息
            lines = result.stdout.split('\n')
            gpu_found = False
            for line in lines:
                if any(gpu in line for gpu in ['Tesla', 'T4', 'P100', 'V100', 'A100']):
                    print(f"   🎯 检测到GPU: {line.strip()}")
                    gpu_found = True

            if not gpu_found:
                print("⚠️  nvidia-smi运行但未检测到GPU")
                return False
            return True
        else:
            print("❌ nvidia-smi失败")
            return False
    except Exception as e:
        print(f"❌ nvidia-smi异常: {e}")
        return False

def install_pytorch_stack():
    """安装PyTorch技术栈"""
    print("\n🔥 安装PyTorch技术栈")
    print("=" * 30)

    # 检查GPU环境
    has_gpu = check_gpu_environment()

    if has_gpu:
        print("\n🎯 检测到GPU，安装CUDA版本PyTorch")
        # GPU环境：优先安装CUDA版本
        pytorch_options = [
            # 选项1: CUDA 12.1版本
            {
                "cmd": "pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121",
                "desc": "PyTorch 2.1.0 CUDA 12.1版本"
            },
            # 选项2: CUDA 11.8版本
            {
                "cmd": "pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118",
                "desc": "PyTorch 2.0.1 CUDA 11.8版本"
            },
            # 选项3: 默认版本
            {
                "cmd": "pip install torch==2.1.0 torchvision==0.16.0",
                "desc": "PyTorch 2.1.0 默认版本"
            }
        ]
    else:
        print("\n💻 未检测到GPU，安装CPU版本PyTorch")
        # CPU环境：安装CPU版本
        pytorch_options = [
            # 选项1: CPU版本
            {
                "cmd": "pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu",
                "desc": "PyTorch 2.1.0 CPU版本"
            },
            # 选项2: 较旧版本
            {
                "cmd": "pip install torch==1.13.1 torchvision==0.14.1",
                "desc": "PyTorch 1.13.1 保守版本"
            }
        ]

    for i, option in enumerate(pytorch_options, 1):
        print(f"\n尝试方案 {i}: {option['desc']}")
        if run_command(option["cmd"], option["desc"]):
            print(f"✅ PyTorch方案 {i} 安装成功")
            break
    else:
        print("❌ 所有PyTorch安装方案都失败")
        return False

    return True

def install_ai_packages():
    """安装AI相关包 - 强制使用兼容版本组合"""
    print("\n🤖 安装AI相关包")
    print("=" * 30)

    # 强制使用经过验证的稳定版本组合 - 确保与原项目完全一致
    # 这些版本经过测试，解决了 cached_download 兼容性问题
    ai_packages = [
        ("huggingface_hub==0.16.4", "HuggingFace Hub"),  # 包含 cached_download，与diffusers兼容
        ("transformers==4.30.2", "Transformers"),        # 稳定版本，支持所有功能
        ("diffusers==0.21.4", "Diffusers"),              # 与 huggingface_hub 0.16.4 完全兼容
        ("accelerate==0.20.3", "Accelerate")             # 稳定版本，支持混合精度训练
    ]

    print("🔧 强制安装兼容版本组合以确保稳定性...")

    success_count = 0
    for package, name in ai_packages:
        # 先尝试强制重装以确保版本正确
        if run_command(f"pip install --force-reinstall {package}", f"强制安装 {name}"):
            success_count += 1
        else:
            # 如果强制重装失败，尝试普通安装
            print(f"   ⚠️  {name} 强制安装失败，尝试普通安装...")
            if run_command(f"pip install {package}", f"安装 {name}"):
                success_count += 1
            else:
                print(f"   ❌ {name} 安装失败")

    print(f"\n📊 AI包安装结果: {success_count}/{len(ai_packages)} 成功")

    # 验证关键兼容性
    print("\n🔍 验证关键兼容性...")
    try:
        from huggingface_hub import cached_download
        print("✅ cached_download 验证成功")
        return True
    except ImportError:
        print("❌ cached_download 仍然不可用")
        print("🔧 执行强力修复...")

        # 强力修复：完全重装关键包
        critical_packages = [
            "huggingface_hub==0.16.4",
            "diffusers==0.21.4"
        ]

        for package in critical_packages:
            print(f"🔄 强力重装 {package}...")
            # 先卸载
            package_name = package.split('==')[0]
            run_command(f"pip uninstall {package_name} -y", f"卸载 {package_name}")
            # 清理缓存
            run_command("pip cache purge", "清理缓存")
            # 重装
            run_command(f"pip install --no-cache-dir {package}", f"重装 {package}")

        # 最终验证
        try:
            # 清理模块缓存
            import sys
            modules_to_clear = ['huggingface_hub', 'diffusers']
            for module in modules_to_clear:
                if module in sys.modules:
                    del sys.modules[module]

            from huggingface_hub import cached_download
            print("✅ 强力修复成功")
            return True
        except ImportError:
            print("❌ 强力修复失败")
            print("💡 建议: 重启内核后重新运行此脚本")
            return False
    except Exception as e:
        print(f"⚠️  其他验证问题: {e}")
        return success_count == len(ai_packages)

def install_utility_packages():
    """安装工具包"""
    print("\n🛠️  安装工具包")
    print("=" * 30)
    
    utility_packages = [
        ("pillow==9.5.0", "Pillow"),
        ("opencv-python==4.8.0.76", "OpenCV"),
        ("matplotlib==3.7.2", "Matplotlib"),
        ("scikit-learn==1.3.0", "Scikit-learn"),
        ("tqdm==4.65.0", "TQDM"),
        ("einops==0.6.1", "Einops"),
        ("packaging>=20.0", "Packaging"),
        ("filelock>=3.0", "FileLock")
    ]
    
    for package, name in utility_packages:
        run_command(f"pip install {package}", f"安装 {name}")

def test_gpu_functionality():
    """测试GPU功能"""
    print("\n🎮 GPU功能测试:")

    try:
        import torch

        # 检查PyTorch版本
        pytorch_version = torch.__version__
        print(f"✅ PyTorch版本: {pytorch_version}")

        # 检查CUDA编译支持
        cuda_version = torch.version.cuda
        print(f"✅ CUDA编译版本: {cuda_version}")

        # 检查CUDA可用性
        cuda_available = torch.cuda.is_available()
        print(f"{'✅' if cuda_available else '❌'} CUDA可用: {cuda_available}")

        if cuda_available:
            # GPU详细信息
            device_count = torch.cuda.device_count()
            print(f"✅ GPU数量: {device_count}")

            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"✅ GPU {i}: {gpu_name}")
                print(f"   内存: {memory_gb:.1f} GB")
                print(f"   计算能力: {props.major}.{props.minor}")

            # 测试GPU操作
            device = torch.device('cuda:0')
            test_tensor = torch.randn(100, 100, device=device)
            result = torch.mm(test_tensor, test_tensor.t())

            print("✅ GPU张量操作成功")
            print(f"   设备: {test_tensor.device}")

            # 内存使用情况
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            print(f"   已分配内存: {memory_allocated:.1f} MB")

            return True
        else:
            # CPU模式
            print("ℹ️  使用CPU模式")
            test_tensor = torch.randn(100, 100)
            result = torch.mm(test_tensor, test_tensor.t())
            print("✅ CPU张量操作成功")

            # 检查是否为CPU版本
            if '+cpu' in pytorch_version:
                print("⚠️  检测到CPU版本PyTorch")
                print("   如需GPU支持，请重新安装CUDA版本")
                return False
            else:
                print("⚠️  CUDA不可用但PyTorch支持CUDA")
                print("   可能是驱动或环境问题")
                return False

    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False

def comprehensive_test():
    """全面测试"""
    print("\n🧪 全面功能测试")
    print("=" * 30)

    # 清理模块缓存
    modules_to_clear = ['numpy', 'torch', 'torchvision', 'diffusers', 'transformers', 'scipy', 'sklearn']
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]

    test_results = {}

    # 测试0: GPU功能
    print("\n0️⃣ 测试GPU功能...")
    gpu_ok = test_gpu_functionality()
    test_results['gpu'] = gpu_ok
    
    # 测试1: NumPy
    print("\n1️⃣ 测试NumPy...")
    try:
        import numpy as np
        test_array = np.random.randn(3, 3)
        result = np.mean(test_array)
        print(f"✅ NumPy {np.__version__}: 功能正常")
        test_results['numpy'] = True
    except Exception as e:
        print(f"❌ NumPy测试失败: {e}")
        test_results['numpy'] = False
    
    # 测试2: PyTorch
    print("\n2️⃣ 测试PyTorch...")
    try:
        import torch
        test_tensor = torch.randn(2, 3)
        result = torch.mean(test_tensor)
        print(f"✅ PyTorch {torch.__version__}: 功能正常")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  使用CPU模式")
        
        test_results['torch'] = True
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        test_results['torch'] = False
    
    # 测试3: TorchVision
    print("\n3️⃣ 测试TorchVision...")
    try:
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}: 导入成功")
        test_results['torchvision'] = True
    except Exception as e:
        print(f"❌ TorchVision测试失败: {e}")
        test_results['torchvision'] = False
    
    # 测试4: Diffusers (关键兼容性测试)
    print("\n4️⃣ 测试Diffusers...")
    try:
        # 首先测试 cached_download 兼容性
        from huggingface_hub import cached_download
        print("✅ cached_download 导入成功")

        import diffusers
        from diffusers import AutoencoderKL, UNet2DConditionModel
        print(f"✅ Diffusers {diffusers.__version__}: 导入成功")
        test_results['diffusers'] = True
    except ImportError as e:
        if 'cached_download' in str(e):
            print(f"❌ Diffusers测试失败: cached_download 兼容性问题")
            print("🔧 这表明需要重新运行环境修复")
            print("💡 建议: 重启内核后重新运行 ultimate_fix_kaggle.py")
        else:
            print(f"❌ Diffusers测试失败: {e}")
        test_results['diffusers'] = False
    except Exception as e:
        print(f"❌ Diffusers测试失败: {e}")
        test_results['diffusers'] = False
    
    # 测试5: VAE功能 (与项目配置一致)
    print("\n5️⃣ 测试VAE功能...")
    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        import torch

        # 创建与项目一致的VAE (128×128 → 32×32)
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[128, 256, 512],
            latent_channels=4,
            sample_size=128,
        )

        # 创建与项目一致的UNet (sample_size=32)
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            cross_attention_dim=768,
        )

        scheduler = DDPMScheduler(num_train_timesteps=1000)

        # 测试完整工作流程
        with torch.no_grad():
            test_input = torch.randn(1, 3, 128, 128)
            test_conditions = torch.randn(1, 1, 768)

            # VAE编码 (128×128 → 32×32)
            latents = vae.encode(test_input).latent_dist.sample()

            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (1,))
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # UNet预测
            pred = unet(noisy_latents, timesteps, encoder_hidden_states=test_conditions, return_dict=False)[0]

            # VAE解码 (32×32 → 128×128)
            reconstructed = vae.decode(latents).sample

        print("✅ VAE+LDM完整工作流程测试通过")
        print(f"   输入: {test_input.shape}")
        print(f"   潜在: {latents.shape}")
        print(f"   重建: {reconstructed.shape}")
        print(f"   UNet预测: {pred.shape}")
        print(f"   压缩比: {test_input.shape[-1] // latents.shape[-1]}倍")
        test_results['vae'] = True

    except Exception as e:
        print(f"❌ VAE功能测试失败: {e}")
        test_results['vae'] = False
    
    # 测试总结
    print("\n📊 测试总结:")
    passed = sum(test_results.values())
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {test_name}")

    print(f"\n总体结果: {passed}/{total} 通过")

    # 关键测试
    critical_tests = ['numpy', 'torch', 'diffusers', 'vae']
    critical_passed = all(test_results.get(test, False) for test in critical_tests)

    # GPU建议
    if not test_results.get('gpu', False):
        print("\n💡 GPU建议:")
        print("   - 检查Kaggle GPU设置")
        print("   - 重启内核后重新运行")
        print("   - 或使用CPU模式训练")

    return critical_passed, test_results

def main():
    """主修复流程"""
    print("🚀 终极Kaggle环境修复工具")
    print("彻底清理并重建所有依赖")
    print("=" * 60)
    
    try:
        # 阶段1: 核弹级清理
        nuclear_cleanup()
        
        # 阶段2: 安装基础系统
        install_base_system()
        
        # 阶段3: 安装PyTorch
        if not install_pytorch_stack():
            print("❌ PyTorch安装失败，无法继续")
            return False
        
        # 阶段4: 安装AI包
        install_ai_packages()
        
        # 阶段5: 安装工具包
        install_utility_packages()
        
        # 阶段6: 全面测试
        print("\n" + "=" * 50 + " 最终测试 " + "=" * 50)
        
        success, test_results = comprehensive_test()

        if success:
            print("\n🎉 修复成功！所有关键组件正常工作")
            print("\n📋 下一步:")

            # 根据GPU状态给出建议
            if test_results.get('gpu', False):
                print("   python train_kaggle.py --stage all")
                print("\n💡 提示:")
                print("   - 环境已完全重建")
                print("   - GPU可用，可以全速训练")
                print("   - 所有版本冲突已解决")
            else:
                print("   python train_kaggle.py --stage all --device cpu")
                print("\n💡 提示:")
                print("   - 环境已完全重建")
                print("   - 使用CPU模式训练")
                print("   - 训练时间会较长")

            return True
        else:
            print("\n⚠️  部分组件仍有问题")
            print("\n🔧 建议:")

            # 具体建议
            if not test_results.get('gpu', False):
                print("   1. 检查Kaggle GPU设置")
                print("   2. 重启内核并重新运行")
            if not test_results.get('torch', False):
                print("   3. PyTorch安装问题")
            if not test_results.get('diffusers', False):
                print("   4. Diffusers版本问题")

            print("   5. 或联系技术支持")
            return False
            
    except Exception as e:
        print(f"\n💥 修复过程中出现异常: {e}")
        print("\n🔧 建议:")
        print("   1. 重启Kaggle内核")
        print("   2. 重新运行此脚本")
        return False

if __name__ == "__main__":
    main()
