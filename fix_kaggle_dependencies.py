#!/usr/bin/env python3
"""
Kaggle依赖冲突修复脚本
解决numpy、JAX、transformers版本冲突问题
"""

import subprocess
import sys
import importlib
import warnings
warnings.filterwarnings("ignore")

def run_pip_command(command, description=""):
    """运行pip命令"""
    print(f"🔄 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} - 成功")
            return True
        else:
            print(f"❌ {description} - 失败")
            if result.stderr:
                print(f"错误: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ {description} - 异常: {e}")
        return False

def check_problematic_packages():
    """检查有问题的包"""
    print("🔍 检查有问题的包...")
    
    problematic = []
    
    # 检查numpy
    try:
        import numpy as np
        if not hasattr(np, 'dtypes'):
            print(f"⚠️  numpy版本问题: {np.__version__} (缺少dtypes属性)")
            problematic.append('numpy')
        else:
            print(f"✅ numpy版本正常: {np.__version__}")
    except ImportError:
        print("❌ numpy未安装")
        problematic.append('numpy')
    
    # 检查JAX
    try:
        import jax
        print(f"⚠️  JAX已安装: {jax.__version__} (可能导致冲突)")
        problematic.append('jax')
    except ImportError:
        print("✅ JAX未安装 (避免冲突)")
    
    # 检查transformers
    try:
        import transformers
        print(f"📦 transformers版本: {transformers.__version__}")
    except ImportError:
        print("❌ transformers未安装")
        problematic.append('transformers')
    
    return problematic

def fix_numpy_jax_conflict():
    """修复numpy和JAX冲突"""
    print("\n🔧 修复numpy和JAX冲突...")
    
    # 方案1: 卸载JAX相关包
    jax_packages = [
        'jax',
        'jaxlib', 
        'flax',
        'optax'
    ]
    
    print("卸载JAX相关包...")
    for package in jax_packages:
        run_pip_command(f"pip uninstall -y {package}", f"卸载 {package}")
    
    # 方案2: 升级numpy到兼容版本
    print("升级numpy...")
    run_pip_command("pip install --upgrade numpy>=1.24.0,<1.26.0", "升级numpy到兼容版本")
    
    return True

def install_core_dependencies():
    """安装核心依赖"""
    print("\n📦 安装核心依赖...")
    
    # 核心包列表 (避免冲突的版本)
    core_packages = [
        "torch==2.1.0",
        "torchvision==0.16.0", 
        "diffusers==0.25.1",
        "transformers==4.36.2",
        "accelerate==0.25.0",
        "numpy>=1.24.0,<1.26.0",
        "pillow>=9.5.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "tqdm>=4.65.0",
        "einops>=0.7.0"
    ]
    
    for package in core_packages:
        run_pip_command(f"pip install {package}", f"安装 {package}")
    
    return True

def test_imports():
    """测试关键导入"""
    print("\n🧪 测试关键导入...")
    
    test_modules = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('diffusers', 'Diffusers'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('scipy', 'SciPy'),
        ('tqdm', 'TQDM'),
        ('einops', 'Einops')
    ]
    
    success_count = 0
    
    for module_name, display_name in test_modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {display_name}: 导入失败 - {e}")
        except Exception as e:
            print(f"⚠️  {display_name}: 导入异常 - {e}")
    
    print(f"\n📊 导入测试: {success_count}/{len(test_modules)} 成功")
    return success_count == len(test_modules)

def test_diffusers_basic():
    """测试Diffusers基本功能"""
    print("\n🔧 测试Diffusers基本功能...")
    
    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        
        # 创建小模型测试
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            sample_size=64,  # 小尺寸测试
        )
        
        unet = UNet2DConditionModel(
            sample_size=8,  # 小尺寸测试
            in_channels=4,
            out_channels=4,
            cross_attention_dim=768,
            layers_per_block=1,  # 减少层数
            block_out_channels=(32, 64),  # 减少通道数
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        )
        
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        print("✅ Diffusers模型创建成功")
        
        # 测试前向传播
        import torch
        
        with torch.no_grad():
            test_input = torch.randn(1, 3, 64, 64)
            latents = vae.encode(test_input).latent_dist.sample()
            reconstructed = vae.decode(latents).sample
            
            print(f"✅ VAE前向传播成功: {test_input.shape} -> {latents.shape} -> {reconstructed.shape}")
            
            # 测试UNet
            timesteps = torch.randint(0, 1000, (1,))
            encoder_hidden_states = torch.randn(1, 1, 768)
            
            noise_pred = unet(
                latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            print(f"✅ UNet前向传播成功: {latents.shape} -> {noise_pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Diffusers测试失败: {e}")
        return False

def create_safe_test_script():
    """创建安全的测试脚本"""
    print("\n📝 创建安全的测试脚本...")
    
    safe_test_content = '''#!/usr/bin/env python3
"""
安全的依赖测试脚本 - 避免导入冲突
"""

def test_core_imports():
    """测试核心导入，避免冲突"""
    import sys
    
    # 测试基础包
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except Exception as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    
    try:
        import diffusers
        print(f"✅ Diffusers: {diffusers.__version__}")
    except Exception as e:
        print(f"❌ Diffusers导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        # 检查dtypes属性
        if hasattr(np, 'dtypes'):
            print("✅ NumPy dtypes属性存在")
        else:
            print("⚠️  NumPy dtypes属性不存在")
    except Exception as e:
        print(f"❌ NumPy导入失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 安全依赖测试")
    if test_core_imports():
        print("✅ 核心依赖测试通过")
    else:
        print("❌ 核心依赖测试失败")
'''
    
    with open('safe_test.py', 'w') as f:
        f.write(safe_test_content)
    
    print("✅ 安全测试脚本已创建: safe_test.py")

def main():
    """主修复流程"""
    print("🔧 Kaggle依赖冲突修复工具")
    print("=" * 50)
    
    # 1. 检查问题
    problematic = check_problematic_packages()
    
    if not problematic:
        print("✅ 未发现依赖问题")
        return
    
    print(f"\n⚠️  发现问题包: {problematic}")
    
    # 2. 修复numpy和JAX冲突
    if 'numpy' in problematic or 'jax' in problematic:
        fix_numpy_jax_conflict()
    
    # 3. 重新安装核心依赖
    install_core_dependencies()
    
    # 4. 测试导入
    print("\n" + "=" * 30 + " 测试阶段 " + "=" * 30)
    
    if test_imports():
        print("✅ 所有包导入成功")
        
        # 5. 测试Diffusers
        if test_diffusers_basic():
            print("✅ Diffusers功能测试通过")
        else:
            print("⚠️  Diffusers功能测试失败")
    else:
        print("❌ 部分包导入失败")
    
    # 6. 创建安全测试脚本
    create_safe_test_script()
    
    print("\n🎉 修复完成！")
    print("\n📋 下一步:")
    print("1. 重启Python内核")
    print("2. 运行: python safe_test.py")
    print("3. 如果测试通过，运行: python train_kaggle.py --stage all")

if __name__ == "__main__":
    main()
