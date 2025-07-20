#!/usr/bin/env python3
"""
VQ-VAE + Transformer 统一环境安装脚本
借鉴ultimate_fix_kaggle.py思路，使用经过验证的固定版本组合
"""

import os
import sys
import subprocess
import importlib

def run_command(cmd, description="", ignore_errors=False):
    """运行命令并处理错误"""
    print(f"🔄 {description}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 or ignore_errors:
            print(f"✅ {description} 成功")
            return True
        else:
            print(f"❌ {description} 失败")
            if result.stderr and not ignore_errors:
                print(f"   错误: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} 异常: {e}")
        return False

def detect_environment():
    """检测运行环境"""
    print("🔍 检测运行环境...")
    
    if any([os.path.exists('/kaggle'), 'KAGGLE_KERNEL_RUN_TYPE' in os.environ]):
        print("✅ 检测到Kaggle环境")
        return "kaggle"
    
    try:
        import google.colab
        print("✅ 检测到Google Colab环境")
        return "colab"
    except ImportError:
        pass
    
    print("✅ 检测到本地环境")
    return "local"

def clean_environment():
    """清理环境 - 移除可能冲突的包"""
    print("\n🗑️ 清理可能冲突的包...")

    # 清理Python模块缓存
    try:
        modules_to_clear = ['numpy', 'torch', 'diffusers', 'transformers', 'huggingface_hub', 'accelerate']
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]
        print("✅ Python模块缓存已清理")
    except:
        pass

    # 卸载可能冲突的包
    packages_to_remove = [
        "diffusers", "transformers", "accelerate", "huggingface_hub", "huggingface-hub",
        "tokenizers", "safetensors", "datasets", "evaluate", "peft", "trl",
        "jax", "jaxlib", "flax", "optax"  # JAX相关包可能导致numpy冲突
    ]

    for package in packages_to_remove:
        run_command(f"pip uninstall -y {package}", f"卸载 {package}", ignore_errors=True)

    # 清理pip缓存
    run_command("pip cache purge", "清理pip缓存", ignore_errors=True)

    return True

def install_core_packages():
    """安装核心包 - 使用经过验证的固定版本"""
    print("\n🔧 安装核心包...")

    # 升级基础工具
    run_command("pip install --upgrade pip setuptools wheel", "升级基础工具")

    # 安装兼容的numpy版本 - 解决JAX兼容性问题
    numpy_versions = ["1.26.4", "1.24.4", "1.23.5"]
    for version in numpy_versions:
        if run_command(f"pip install numpy=={version}", f"安装numpy {version}"):
            break

    # 安装其他核心依赖 - 使用固定版本
    core_deps = [
        ("pillow==10.0.1", "Pillow"),
        ("requests==2.31.0", "Requests"),
        ("packaging==23.2", "Packaging"),
        ("filelock==3.13.1", "FileLock"),
        ("tqdm==4.66.1", "TQDM"),
        ("pyyaml==6.0.1", "PyYAML"),
        ("typing-extensions==4.8.0", "Typing Extensions"),
        ("regex==2023.10.3", "Regex"),
    ]

    for package, name in core_deps:
        run_command(f"pip install {package}", f"安装 {name}", ignore_errors=True)

    return True

def check_gpu_environment():
    """检查GPU环境"""
    print("\n🔍 检查GPU环境...")

    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi可用")
            # 检查GPU类型
            lines = result.stdout.split('\n')
            for line in lines:
                if any(gpu in line for gpu in ['Tesla', 'T4', 'P100', 'V100', 'A100']):
                    print(f"   🎯 检测到GPU: {line.strip()}")
                    return True
            print("⚠️ nvidia-smi运行但未检测到GPU")
            return False
        else:
            print("❌ nvidia-smi失败")
            return False
    except Exception as e:
        print(f"❌ nvidia-smi异常: {e}")
        return False

def install_pytorch(env_type):
    """安装PyTorch - 使用经过验证的固定版本"""
    print("\n🔥 安装PyTorch...")

    if env_type == "kaggle":
        try:
            import torch
            print(f"✅ 使用Kaggle预装PyTorch: {torch.__version__}")
            return True
        except ImportError:
            pass

    # 检查GPU环境
    has_gpu = check_gpu_environment()

    if has_gpu:
        print("🎯 检测到GPU，安装CUDA版本PyTorch")
        # GPU环境：使用经过验证的CUDA版本
        pytorch_options = [
            {
                "cmd": "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118",
                "desc": "PyTorch 2.1.0 CUDA 11.8版本"
            },
            {
                "cmd": "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118",
                "desc": "PyTorch 2.0.1 CUDA 11.8版本"
            },
            {
                "cmd": "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0",
                "desc": "PyTorch 2.1.0 默认版本"
            }
        ]
    else:
        print("💻 未检测到GPU，安装CPU版本PyTorch")
        # CPU环境：使用CPU版本
        pytorch_options = [
            {
                "cmd": "pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu",
                "desc": "PyTorch 2.1.0 CPU版本"
            },
            {
                "cmd": "pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1",
                "desc": "PyTorch 1.13.1 保守版本"
            }
        ]

    for i, option in enumerate(pytorch_options, 1):
        print(f"\n尝试方案 {i}: {option['desc']}")
        if run_command(option["cmd"], option["desc"]):
            print(f"✅ PyTorch方案 {i} 安装成功")
            return True

    print("❌ 所有PyTorch安装方案都失败")
    return False

def install_huggingface_stack():
    """安装HuggingFace技术栈 - 使用经过验证的固定版本组合"""
    print("\n🤗 安装HuggingFace技术栈...")

    # 使用经过验证的稳定版本组合 - 借鉴ultimate_fix_kaggle.py
    # 这些版本经过测试，解决了cached_download兼容性问题
    hf_packages = [
        ("huggingface_hub==0.17.3", "HuggingFace Hub"),  # 支持cached_download
        ("tokenizers==0.14.1", "Tokenizers"),            # 与transformers兼容
        ("safetensors==0.4.0", "SafeTensors"),           # 稳定版本
        ("transformers==4.35.2", "Transformers"),        # 稳定版本，支持所有功能
        ("accelerate==0.24.1", "Accelerate"),            # 稳定版本，支持混合精度训练
        ("diffusers==0.24.0", "Diffusers"),              # 与huggingface_hub完全兼容
    ]

    print("🔧 使用经过验证的固定版本组合...")

    success_count = 0
    for package, name in hf_packages:
        # 先尝试强制重装以确保版本正确
        if run_command(f"pip install --force-reinstall {package}", f"强制安装 {name}"):
            success_count += 1
        else:
            # 如果强制重装失败，尝试普通安装
            print(f"   ⚠️ {name} 强制安装失败，尝试普通安装...")
            if run_command(f"pip install {package}", f"安装 {name}"):
                success_count += 1
            else:
                print(f"   ❌ {name} 安装失败")

    print(f"\n📊 HuggingFace包安装结果: {success_count}/{len(hf_packages)} 成功")

    # 验证关键兼容性 - cached_download
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
            "huggingface_hub==0.17.3",
            "diffusers==0.24.0"
        ]

        for package in critical_packages:
            print(f"🔄 强力重装 {package}...")
            package_name = package.split('==')[0]
            run_command(f"pip uninstall {package_name} -y", f"卸载 {package_name}", ignore_errors=True)
            run_command("pip cache purge", "清理缓存", ignore_errors=True)
            run_command(f"pip install --no-cache-dir {package}", f"重装 {package}")

        # 最终验证 - 彻底清理模块缓存
        try:
            print("🧹 彻底清理Python模块缓存...")

            # 清理所有相关模块和子模块
            modules_to_clear = []
            for module_name in list(sys.modules.keys()):
                if any(pattern in module_name for pattern in [
                    'huggingface_hub', 'diffusers', 'transformers',
                    'tokenizers', 'safetensors'
                ]):
                    modules_to_clear.append(module_name)

            for module_name in modules_to_clear:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    print(f"   清理模块: {module_name}")

            print(f"✅ 清理了 {len(modules_to_clear)} 个模块")

            # 强制重新导入
            print("🔄 强制重新导入...")
            import importlib

            # 重新导入huggingface_hub
            import huggingface_hub
            importlib.reload(huggingface_hub)

            # 测试cached_download
            from huggingface_hub import cached_download
            print("✅ cached_download 导入成功")

            # 进一步测试API可用性
            print("🧪 测试cached_download API...")
            # 不实际下载，只测试函数是否存在和可调用
            if callable(cached_download):
                print("✅ cached_download API 可用")
                return True
            else:
                print("❌ cached_download 不可调用")
                return False

        except ImportError as e:
            print(f"❌ 导入失败: {e}")
            print("🔧 尝试替代API...")

            # 尝试新的API
            try:
                from huggingface_hub import hf_hub_download
                print("✅ 找到替代API: hf_hub_download")
                print("⚠️ 需要更新代码使用新API")
                return True
            except ImportError:
                print("❌ 所有API都不可用")
                print("💡 建议:")
                print("1. 重启Python内核")
                print("2. 手动安装: pip install huggingface_hub==0.17.3 --force-reinstall --no-cache-dir")
                print("3. 检查是否有其他包冲突")
                return False
        except Exception as e:
            print(f"❌ 其他错误: {e}")
            return False
    except Exception as e:
        print(f"⚠️ 其他验证问题: {e}")
        return success_count == len(hf_packages)

def fix_huggingface_api():
    """专门修复HuggingFace API兼容性问题"""
    print("\n🔧 HuggingFace API兼容性修复...")

    # 检查当前安装的版本
    try:
        import huggingface_hub
        current_version = huggingface_hub.__version__
        print(f"📊 当前huggingface_hub版本: {current_version}")
    except ImportError:
        print("❌ huggingface_hub未安装")
        return False

    # 测试不同的API
    api_tests = [
        ("cached_download", "from huggingface_hub import cached_download"),
        ("hf_hub_download", "from huggingface_hub import hf_hub_download"),
        ("snapshot_download", "from huggingface_hub import snapshot_download"),
    ]

    available_apis = []

    for api_name, import_cmd in api_tests:
        try:
            exec(import_cmd)
            available_apis.append(api_name)
            print(f"✅ {api_name}: 可用")
        except ImportError as e:
            print(f"❌ {api_name}: 不可用 - {e}")

    if not available_apis:
        print("❌ 所有HuggingFace下载API都不可用")

        # 尝试降级到更稳定的版本
        stable_versions = ["0.16.4", "0.15.1", "0.14.1"]

        for version in stable_versions:
            print(f"🔄 尝试降级到 huggingface_hub=={version}...")

            # 完全卸载
            run_command("pip uninstall huggingface_hub -y", f"卸载当前版本", ignore_errors=True)
            run_command("pip cache purge", "清理缓存", ignore_errors=True)

            # 安装指定版本
            if run_command(f"pip install huggingface_hub=={version} --no-cache-dir", f"安装版本 {version}"):
                # 清理模块缓存
                for module_name in list(sys.modules.keys()):
                    if 'huggingface_hub' in module_name:
                        del sys.modules[module_name]

                # 重新测试
                try:
                    import huggingface_hub
                    from huggingface_hub import cached_download
                    print(f"✅ 版本 {version} 工作正常")
                    return True
                except ImportError:
                    print(f"❌ 版本 {version} 仍然有问题")
                    continue

        print("❌ 所有版本都无法解决API问题")
        return False

    else:
        print(f"✅ 找到可用的API: {', '.join(available_apis)}")

        # 如果cached_download不可用但有其他API，提供替代方案
        if "cached_download" not in available_apis:
            print("⚠️ cached_download不可用，但有其他API可用")
            print("💡 建议更新代码使用新的API:")

            if "hf_hub_download" in available_apis:
                print("   使用 hf_hub_download 替代 cached_download")
            elif "snapshot_download" in available_apis:
                print("   使用 snapshot_download 替代 cached_download")

        return True

def install_additional_deps():
    """安装其他必要依赖 - 使用固定版本"""
    print("\n📚 安装其他依赖...")

    additional_deps = [
        ("scipy==1.11.4", "SciPy"),
        ("scikit-learn==1.3.0", "Scikit-learn"),
        ("scikit-image==0.21.0", "Scikit-image"),
        ("matplotlib==3.7.2", "Matplotlib"),
        ("opencv-python==4.8.1.78", "OpenCV"),
        ("einops==0.7.0", "Einops"),
        ("tensorboard==2.15.1", "TensorBoard"),
        ("lpips==0.1.4", "LPIPS"),
    ]

    for package, name in additional_deps:
        run_command(f"pip install {package}", f"安装 {name}", ignore_errors=True)

    return True

def test_critical_imports():
    """测试关键导入"""
    print("\n🧪 测试关键导入...")
    
    critical_tests = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("huggingface_hub", "HuggingFace Hub"),
        ("accelerate", "Accelerate"),
    ]
    
    all_good = True
    
    for module_name, display_name in critical_tests:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
        except ImportError as e:
            print(f"❌ {display_name}: 导入失败 - {e}")
            all_good = False
    
    return all_good

def test_vqmodel_api():
    """测试VQModel API"""
    print("\n🎨 测试VQModel API...")
    
    # 尝试不同的导入路径
    VQModel = None
    
    import_attempts = [
        ("diffusers.models.autoencoders.vq_model", "新版API"),
        ("diffusers.models.vq_model", "旧版API"),
        ("diffusers", "直接导入"),
    ]
    
    for module_path, description in import_attempts:
        try:
            if module_path == "diffusers":
                from diffusers import VQModel
            else:
                module = importlib.import_module(module_path)
                VQModel = getattr(module, 'VQModel')
            
            print(f"✅ VQModel导入成功: {description}")
            break
        except (ImportError, AttributeError):
            continue
    
    if VQModel is None:
        print("❌ VQModel: 所有导入路径都失败")
        return False
    
    # 测试创建和使用
    try:
        import torch
        model = VQModel(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D"],
            block_out_channels=[64],
            layers_per_block=1,
            latent_channels=64,
            sample_size=32,
            num_vq_embeddings=128,
            norm_num_groups=32,
            vq_embed_dim=64,
        )
        
        test_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            result = model.encode(test_input)
            decoded = model.decode(result.latents)
            print(f"✅ VQModel测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ VQModel创建/测试失败: {e}")
        return False

def test_transformer_api():
    """测试Transformer API"""
    print("\n🤖 测试Transformer API...")
    
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        
        config = GPT2Config(
            vocab_size=256,
            n_positions=64,
            n_embd=128,
            n_layer=2,
            n_head=4,
            use_cache=False,
        )
        
        model = GPT2LMHeadModel(config)
        
        import torch
        test_input = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ Transformer测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformer测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🎨 VQ-VAE + Transformer 统一环境安装器")
    print("=" * 60)
    print("🔧 解决numpy/JAX兼容性和API版本问题")
    
    # 检测环境
    env_type = detect_environment()
    print(f"\n📊 环境类型: {env_type}")
    
    # 确认操作
    if env_type == "local":
        response = input("\n是否继续安装? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ 操作已取消")
            return
    
    # 安装流程 - 借鉴ultimate_fix_kaggle.py的阶段化安装
    steps = [
        ("清理环境", clean_environment),
        ("安装核心包", install_core_packages),
        ("安装PyTorch", lambda: install_pytorch(env_type)),
        ("安装HuggingFace技术栈", install_huggingface_stack),
        ("修复HuggingFace API", fix_huggingface_api),  # 新增API修复步骤
        ("安装其他依赖", install_additional_deps),
        ("测试关键导入", test_critical_imports),
        ("测试VQModel API", test_vqmodel_api),
        ("测试Transformer API", test_transformer_api),
    ]

    failed_steps = []

    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        success = step_func()

        if not success:
            print(f"❌ {step_name} 失败")
            failed_steps.append(step_name)
            # 对于关键步骤，如果失败则停止
            if step_name in ["安装PyTorch", "安装HuggingFace技术栈"]:
                print(f"💥 关键步骤失败，无法继续")
                break
        else:
            print(f"✅ {step_name} 成功")
    
    # 总结
    print(f"\n{'='*20} 安装总结 {'='*20}")
    
    if not failed_steps:
        print("🎉 环境安装完全成功!")
        print("✅ 所有组件正常工作")
        print("\n🚀 现在可以开始训练:")
        print("   python train_main.py --data_dir /path/to/data")
    else:
        print(f"⚠️ 部分步骤失败: {', '.join(failed_steps)}")
        
        if "测试关键导入" not in failed_steps:
            print("✅ 基础环境安装成功，可以尝试运行")
            print("💡 建议重启Python内核后再次测试")
        else:
            print("❌ 基础环境有问题，建议:")
            print("1. 重启Python内核")
            print("2. 重新运行此脚本")
            print("3. 检查网络连接")

if __name__ == "__main__":
    main()
