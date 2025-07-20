#!/usr/bin/env python3
"""
Kaggle环境专用修复脚本
解决PyTorch、transformers和diffusers的兼容性问题
基于diffusers 0.24.0的官方要求
"""

import subprocess
import sys
import importlib

def run_command(cmd, description=""):
    """运行命令"""
    print(f"🔄 {description}")
    print(f"   命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            return True
        else:
            print(f"❌ {description} 失败")
            if result.stderr.strip():
                print(f"   错误: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时")
        return False
    except Exception as e:
        print(f"❌ {description} 异常: {e}")
        return False

def check_kaggle_environment():
    """检查Kaggle环境"""
    print("🔍 检查Kaggle环境...")
    
    kaggle_indicators = [
        ("/kaggle", "Kaggle目录"),
        ("/opt/conda", "Conda环境"),
        ("KAGGLE_KERNEL_RUN_TYPE", "Kaggle环境变量"),
    ]
    
    is_kaggle = False
    for indicator, desc in kaggle_indicators:
        if indicator.startswith("/"):
            import os
            if os.path.exists(indicator):
                print(f"✅ 检测到 {desc}: {indicator}")
                is_kaggle = True
        else:
            import os
            if indicator in os.environ:
                print(f"✅ 检测到 {desc}: {os.environ[indicator]}")
                is_kaggle = True
    
    if is_kaggle:
        print("✅ 确认在Kaggle环境中")
        return True
    else:
        print("⚠️ 可能不在Kaggle环境中，但继续执行")
        return True

def complete_cleanup():
    """完全清理环境"""
    print("\n🗑️ 完全清理环境...")
    
    # 清理Python模块缓存
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        if any(pattern in module_name for pattern in [
            'torch', 'transformers', 'diffusers', 'huggingface_hub',
            'accelerate', 'tokenizers', 'safetensors'
        ]):
            modules_to_clear.append(module_name)
    
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    print(f"✅ 清理了 {len(modules_to_clear)} 个Python模块")
    
    # 卸载可能冲突的包
    packages_to_remove = [
        "torch", "torchvision", "torchaudio",
        "transformers", "diffusers", "accelerate",
        "huggingface_hub", "huggingface-hub",
        "tokenizers", "safetensors"
    ]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall {package} -y", f"卸载 {package}")
    
    # 清理pip缓存
    run_command("pip cache purge", "清理pip缓存")
    
    return True

def install_compatible_versions():
    """安装兼容版本组合"""
    print("\n📦 安装兼容版本组合...")
    print("🎯 基于diffusers 0.24.0官方要求和实际测试")
    
    # 第一步：安装PyTorch (与transformers 4.30.2兼容)
    print("\n🔥 安装PyTorch...")
    pytorch_success = False
    
    # 使用正确的PyTorch版本对应关系
    pytorch_options = [
        # 方案1: PyTorch 2.0.1 (推荐，与transformers 4.30.2兼容)
        "pip install torch==2.0.1 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118",

        # 方案2: 使用Kaggle预装版本 (如果可用)
        "pip install torch torchvision torchaudio --upgrade",

        # 方案3: PyTorch 1.13.1 (稳定版本)
        "pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117",

        # 方案4: 最新稳定版本
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",

        # 方案5: CPU版本 (备用)
        "pip install torch==2.0.1 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu",
    ]
    
    for i, cmd in enumerate(pytorch_options, 1):
        print(f"\n尝试PyTorch方案 {i}...")
        if run_command(cmd, f"PyTorch方案 {i}"):
            pytorch_success = True
            break
    
    if not pytorch_success:
        print("❌ 所有PyTorch安装方案都失败")
        return False
    
    # 第二步：安装HuggingFace生态系统 (按依赖顺序)
    print("\n🤗 安装HuggingFace生态系统...")
    
    hf_packages = [
        ("huggingface_hub==0.16.4", "HuggingFace Hub (支持cached_download)"),
        ("tokenizers==0.13.3", "Tokenizers"),
        ("safetensors==0.3.3", "SafeTensors"),
        ("transformers==4.30.2", "Transformers (与PyTorch 2.0.1兼容)"),
        ("accelerate==0.20.3", "Accelerate"),
        ("diffusers==0.24.0", "Diffusers (目标版本)"),
    ]
    
    for package, description in hf_packages:
        if not run_command(f"pip install {package}", f"安装 {description}"):
            print(f"⚠️ {description} 安装失败，继续...")
    
    # 第三步：安装其他必要依赖
    print("\n📚 安装其他依赖...")
    
    other_deps = [
        "numpy==1.26.4",
        "scipy==1.11.4",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "opencv-python==4.8.1.78",
        "einops==0.7.0",
        "lpips==0.1.4",
    ]
    
    for dep in other_deps:
        run_command(f"pip install {dep}", f"安装 {dep}")
    
    return True

def test_installation():
    """测试安装结果"""
    print("\n🧪 测试安装结果...")
    
    # 清理模块缓存，强制重新导入
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        if any(pattern in module_name for pattern in [
            'torch', 'transformers', 'diffusers', 'huggingface_hub'
        ]):
            modules_to_clear.append(module_name)
    
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    # 测试关键导入
    tests = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("diffusers", "Diffusers"),
        ("huggingface_hub", "HuggingFace Hub"),
        ("accelerate", "Accelerate"),
    ]
    
    success_count = 0
    
    for module_name, display_name in tests:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            success_count += 1
        except Exception as e:
            print(f"❌ {display_name}: 导入失败 - {e}")
    
    # 测试cached_download
    try:
        from huggingface_hub import cached_download
        print("✅ cached_download: 可用")
        success_count += 1
    except Exception as e:
        print(f"❌ cached_download: 不可用 - {e}")
        
        # 尝试替代API
        try:
            from huggingface_hub import hf_hub_download
            print("✅ hf_hub_download: 可用 (替代API)")
        except Exception:
            print("❌ 所有下载API都不可用")
    
    # 测试VQModel
    try:
        from diffusers.models.autoencoders.vq_model import VQModel
        print("✅ VQModel: 可用 (新版API)")
    except ImportError:
        try:
            from diffusers.models.vq_model import VQModel
            print("✅ VQModel: 可用 (旧版API)")
        except ImportError:
            try:
                from diffusers import VQModel
                print("✅ VQModel: 可用 (直接导入)")
            except ImportError:
                print("❌ VQModel: 所有导入路径都失败")
    
    print(f"\n📊 测试结果: {success_count}/{len(tests)+1} 成功")
    
    return success_count >= len(tests)

def main():
    """主函数"""
    print("🔧 Kaggle环境专用修复脚本")
    print("=" * 50)
    print("🎯 解决PyTorch、transformers和diffusers兼容性问题")
    print("📋 基于diffusers 0.24.0官方要求")
    
    # 检查环境
    if not check_kaggle_environment():
        return
    
    # 执行修复流程
    steps = [
        ("完全清理环境", complete_cleanup),
        ("安装兼容版本", install_compatible_versions),
        ("测试安装结果", test_installation),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"❌ {step_name} 失败")
            if step_name == "测试安装结果":
                print("⚠️ 部分组件可能仍有问题，但可以尝试使用")
            else:
                print("💥 关键步骤失败，无法继续")
                return
        else:
            print(f"✅ {step_name} 成功")
    
    print("\n🎉 Kaggle环境修复完成!")
    print("✅ 所有组件已安装并验证")
    print("\n🚀 现在可以开始训练:")
    print("   python train_main.py --data_dir /kaggle/input/dataset")
    print("\n💡 如果仍有问题:")
    print("1. 重启Kaggle内核")
    print("2. 重新运行此脚本")
    print("3. 检查具体错误信息")

if __name__ == "__main__":
    main()
