#!/usr/bin/env python3
"""
依赖和API兼容性测试脚本
验证所有库版本和API是否兼容
"""

import sys
import importlib
import subprocess
from packaging import version
import warnings
warnings.filterwarnings("ignore")

# 必需的依赖版本
REQUIRED_VERSIONS = {
    'torch': '2.0.0',
    'torchvision': '0.15.0',
    'diffusers': '0.25.0',
    'transformers': '4.35.0',
    'accelerate': '0.24.0',
    'PIL': '9.5.0',  # Pillow
    'cv2': '4.8.0',  # opencv-python
    'matplotlib': '3.7.0',
    'skimage': '0.21.0',  # scikit-image
    'sklearn': '1.3.0',  # scikit-learn
    'scipy': '1.11.0',
    'numpy': '1.24.0',
    'tqdm': '4.65.0',
    'einops': '0.7.0'
}

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    python_version = sys.version_info
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Python版本过低: {python_version.major}.{python_version.minor}")
        print("   需要Python 3.8或更高版本")
        return False
    else:
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        return True

def get_package_version(package_name, import_name=None):
    """获取包版本，避免导入冲突"""
    if import_name is None:
        import_name = package_name

    # 有冲突的包使用pkg_resources
    problematic_packages = ['torchvision', 'transformers']

    if package_name in problematic_packages:
        try:
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
        except Exception as e:
            print(f"⚠️  无法获取 {package_name} 版本: {e}")
            return "unknown"

    try:
        module = importlib.import_module(import_name)

        # 尝试不同的版本属性
        for attr in ['__version__', 'version', 'VERSION']:
            if hasattr(module, attr):
                return getattr(module, attr)

        # 特殊处理
        if package_name == 'PIL':
            return module.PILLOW_VERSION
        elif package_name == 'cv2':
            return module.__version__
        elif package_name == 'sklearn':
            return module.__version__
        elif package_name == 'skimage':
            return module.__version__

        return "unknown"

    except ImportError:
        return None
    except Exception as e:
        print(f"⚠️  导入 {import_name} 时出错: {e}")
        # 尝试备用方法
        try:
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
        except:
            return "error"

def check_package_versions():
    """检查所有包版本"""
    print("\n📦 检查包版本...")
    
    # 包名映射
    import_mapping = {
        'PIL': 'PIL',
        'cv2': 'cv2', 
        'sklearn': 'sklearn',
        'skimage': 'skimage'
    }
    
    results = {}
    all_good = True
    
    for package, required_ver in REQUIRED_VERSIONS.items():
        import_name = import_mapping.get(package, package)
        current_ver = get_package_version(package, import_name)
        
        if current_ver is None:
            print(f"❌ {package}: 未安装")
            results[package] = {'status': 'missing', 'current': None, 'required': required_ver}
            all_good = False
        elif current_ver == "unknown":
            print(f"⚠️  {package}: 已安装但无法获取版本")
            results[package] = {'status': 'unknown', 'current': 'unknown', 'required': required_ver}
        else:
            try:
                if version.parse(current_ver) >= version.parse(required_ver):
                    print(f"✅ {package}: {current_ver} (>= {required_ver})")
                    results[package] = {'status': 'ok', 'current': current_ver, 'required': required_ver}
                else:
                    print(f"⚠️  {package}: {current_ver} (需要 >= {required_ver})")
                    results[package] = {'status': 'outdated', 'current': current_ver, 'required': required_ver}
                    all_good = False
            except Exception as e:
                print(f"⚠️  {package}: {current_ver} (版本比较失败: {e})")
                results[package] = {'status': 'unknown', 'current': current_ver, 'required': required_ver}
    
    return results, all_good

def test_diffusers_api():
    """测试Diffusers API兼容性"""
    print("\n🔧 测试Diffusers API兼容性...")
    
    try:
        # 测试AutoencoderKL
        from diffusers import AutoencoderKL
        print("✅ AutoencoderKL 导入成功")
        
        # 测试创建VAE
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            sample_size=256,
        )
        print("✅ AutoencoderKL 创建成功")
        
        # 测试UNet2DConditionModel
        from diffusers import UNet2DConditionModel
        print("✅ UNet2DConditionModel 导入成功")
        
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            cross_attention_dim=768,
        )
        print("✅ UNet2DConditionModel 创建成功")
        
        # 测试调度器
        from diffusers import DDPMScheduler, DDIMScheduler
        print("✅ 调度器导入成功")
        
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )
        print("✅ DDPMScheduler 创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ Diffusers API测试失败: {e}")
        return False

def test_torch_functionality():
    """测试PyTorch功能"""
    print("\n🔥 测试PyTorch功能...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        # 检查CUDA
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ CUDA可用: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            device = torch.device("cpu")
            print("⚠️  CUDA不可用，使用CPU")
        
        # 测试基本操作
        x = torch.randn(2, 3, 64, 64).to(device)
        conv = nn.Conv2d(3, 16, 3, padding=1).to(device)
        y = conv(x)
        print(f"✅ 基本张量操作成功: {x.shape} -> {y.shape}")
        
        # 测试混合精度
        try:
            from torch.cuda.amp import autocast, GradScaler
            with autocast():
                y = conv(x)
            print("✅ 混合精度支持")
        except:
            print("⚠️  混合精度不支持")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        return False

def test_data_loading():
    """测试数据加载功能"""
    print("\n📊 测试数据加载功能...")
    
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from PIL import Image
        import numpy as np
        import torchvision.transforms as transforms
        
        # 创建测试数据集
        class TestDataset(Dataset):
            def __init__(self):
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor()
                ])
            
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                # 创建假图像
                img = Image.new('RGB', (256, 256), (idx*25, 100, 150))
                return {
                    'image': self.transform(img),
                    'user_id': idx % 5,
                    'user_idx': idx % 5
                }
        
        dataset = TestDataset()
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 测试一个batch
        for batch in dataloader:
            assert batch['image'].shape == (2, 3, 256, 256)
            assert len(batch['user_id']) == 2
            break
        
        print("✅ 数据加载测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False

def test_training_components():
    """测试训练组件"""
    print("\n🏋️ 测试训练组件...")
    
    try:
        import torch
        import torch.nn as nn
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        
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
        
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        # 测试前向传播
        with torch.no_grad():
            # VAE测试
            test_images = torch.randn(1, 3, 256, 256).to(device)
            posterior = vae.encode(test_images).latent_dist
            latents = posterior.sample()
            reconstructed = vae.decode(latents).sample
            
            print(f"✅ VAE前向传播: {test_images.shape} -> {latents.shape} -> {reconstructed.shape}")
            
            # UNet测试
            timesteps = torch.randint(0, 1000, (1,)).to(device)
            encoder_hidden_states = torch.randn(1, 1, 768).to(device)
            
            noise_pred = unet(
                latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            print(f"✅ UNet前向传播: {latents.shape} -> {noise_pred.shape}")
        
        # 测试优化器
        optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4)
        print("✅ 优化器创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练组件测试失败: {e}")
        return False

def test_accelerate():
    """测试Accelerate库"""
    print("\n🚀 测试Accelerate库...")
    
    try:
        from accelerate import Accelerator
        
        accelerator = Accelerator()
        print(f"✅ Accelerator创建成功 (设备: {accelerator.device})")
        
        # 测试模型准备
        import torch.nn as nn
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        model, optimizer = accelerator.prepare(model, optimizer)
        print("✅ 模型和优化器准备成功")
        
        return True
        
    except Exception as e:
        print(f"❌ Accelerate测试失败: {e}")
        return False

def install_missing_packages(results):
    """安装缺失的包"""
    print("\n📥 安装缺失或过时的包...")
    
    to_install = []
    for package, info in results.items():
        if info['status'] in ['missing', 'outdated']:
            if package == 'PIL':
                to_install.append('Pillow>=9.5.0')
            elif package == 'cv2':
                to_install.append('opencv-python>=4.8.0')
            elif package == 'sklearn':
                to_install.append('scikit-learn>=1.3.0')
            elif package == 'skimage':
                to_install.append('scikit-image>=0.21.0')
            else:
                to_install.append(f"{package}>={info['required']}")
    
    if to_install:
        print(f"需要安装: {', '.join(to_install)}")
        
        response = input("是否自动安装? (y/n): ")
        if response.lower() == 'y':
            for package in to_install:
                print(f"安装 {package}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"✅ {package} 安装成功")
                except subprocess.CalledProcessError as e:
                    print(f"❌ {package} 安装失败: {e}")
        else:
            print("跳过自动安装")
    else:
        print("✅ 所有包都已正确安装")

def main():
    """主测试函数"""
    print("🧪 微多普勒VAE项目 - 依赖和API兼容性测试")
    print("=" * 60)
    
    all_tests_passed = True
    
    # 1. Python版本检查
    if not check_python_version():
        all_tests_passed = False
    
    # 2. 包版本检查
    results, versions_ok = check_package_versions()
    if not versions_ok:
        all_tests_passed = False
        install_missing_packages(results)
        
        # 重新检查
        print("\n🔄 重新检查包版本...")
        results, versions_ok = check_package_versions()
    
    # 3. API兼容性测试
    if not test_diffusers_api():
        all_tests_passed = False
    
    if not test_torch_functionality():
        all_tests_passed = False
    
    if not test_data_loading():
        all_tests_passed = False
    
    if not test_training_components():
        all_tests_passed = False
    
    if not test_accelerate():
        all_tests_passed = False
    
    # 总结
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 所有测试通过！环境已准备就绪")
        print("✅ 可以安全地开始训练")
        
        print("\n📋 下一步:")
        print("1. python kaggle_config.py  # 验证数据集")
        print("2. python train_kaggle.py --stage all  # 开始训练")
        
    else:
        print("❌ 部分测试失败")
        print("⚠️  请解决上述问题后再开始训练")
        
        print("\n🔧 常见解决方案:")
        print("1. 升级pip: pip install --upgrade pip")
        print("2. 安装/升级包: pip install -r requirements.txt")
        print("3. 重启Python环境")
    
    return all_tests_passed

if __name__ == "__main__":
    main()
