#!/usr/bin/env python3
"""
依赖冲突修复脚本
专门解决NumPy版本冲突和torchao依赖问题
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """运行命令并返回是否成功"""
    print(f"🔄 {description}")
    print(f"   命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            return True
        else:
            print(f"❌ {description} 失败")
            if result.stderr:
                print(f"   错误: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时")
        return False
    except Exception as e:
        print(f"❌ {description} 异常: {e}")
        return False

def nuclear_cleanup():
    """核弹级清理"""
    print("💥 核弹级清理...")
    
    # 卸载所有相关包
    packages_to_remove = [
        "torch", "torchvision", "torchaudio", "torchao",
        "diffusers", "transformers", "accelerate", 
        "huggingface_hub", "tokenizers", "safetensors",
        "numpy", "scipy", "scikit-learn"
    ]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall {package} -y", f"卸载 {package}")
    
    # 清理缓存
    run_command("pip cache purge", "清理pip缓存")
    
    # 清理Python模块缓存
    for module_name in list(sys.modules.keys()):
        if any(pattern in module_name.lower() for pattern in [
            'torch', 'diffusers', 'transformers', 'numpy', 'scipy'
        ]):
            if module_name in sys.modules:
                del sys.modules[module_name]
    
    print("✅ 核弹级清理完成")

def install_pytorch_stable():
    """安装稳定版PyTorch"""
    print("🔥 安装稳定版PyTorch...")
    
    # 使用稳定版本，避免torchao依赖问题
    pytorch_options = [
        "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118",
        "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
    ]
    
    for cmd in pytorch_options:
        if run_command(cmd, "安装PyTorch"):
            return True
    
    return False

def install_numpy_compatible():
    """安装兼容的NumPy版本"""
    print("🔢 安装兼容的NumPy版本...")
    
    # 强制安装NumPy 1.x版本
    return run_command("pip install 'numpy>=1.21.0,<2.0' --force-reinstall", "安装NumPy 1.x")

def install_diffusers_minimal():
    """安装最小化diffusers"""
    print("🎨 安装最小化diffusers...")
    
    # 尝试不同的稳定版本
    diffusers_versions = [
        "diffusers==0.27.2",
        "diffusers==0.26.3", 
        "diffusers==0.25.1",
    ]
    
    for version in diffusers_versions:
        print(f"\n尝试 {version}...")
        if run_command(f"pip install {version} --no-deps", f"安装 {version} (无依赖)"):
            # 手动安装必要依赖
            deps = [
                "requests", "filelock", "importlib_metadata",
                "Pillow", "regex", "tqdm"
            ]
            
            all_deps_ok = True
            for dep in deps:
                if not run_command(f"pip install {dep}", f"安装依赖 {dep}"):
                    all_deps_ok = False
            
            if all_deps_ok:
                # 测试导入
                if test_diffusers_import():
                    print(f"✅ {version} 安装成功")
                    return True
    
    return False

def install_transformers_minimal():
    """安装最小化transformers"""
    print("🤖 安装最小化transformers...")
    
    # 使用较早的稳定版本
    transformers_versions = [
        "transformers==4.35.2",
        "transformers==4.30.2",
        "transformers==4.25.1",
    ]
    
    for version in transformers_versions:
        if run_command(f"pip install {version}", f"安装 {version}"):
            if test_transformers_import():
                print(f"✅ {version} 安装成功")
                return True
    
    return False

def test_diffusers_import():
    """测试diffusers导入"""
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "from diffusers.models.autoencoders.vq_model import VQModel; print('SUCCESS')"
        ], capture_output=True, text=True, timeout=30)
        
        return result.returncode == 0 and "SUCCESS" in result.stdout
    except:
        return False

def test_transformers_import():
    """测试transformers导入"""
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "from transformers import GPT2Config; print('SUCCESS')"
        ], capture_output=True, text=True, timeout=30)
        
        return result.returncode == 0 and "SUCCESS" in result.stdout
    except:
        return False

def install_additional_safe():
    """安装其他安全依赖"""
    print("📚 安装其他安全依赖...")
    
    safe_packages = [
        "safetensors==0.3.3",
        "accelerate==0.21.0", 
        "tokenizers==0.13.3",
        "huggingface_hub==0.16.4",
        "einops==0.6.1",
        "matplotlib==3.7.2",
        "opencv-python==4.8.1.78",
        "scikit-learn==1.3.0",
        "scipy==1.11.4",
    ]
    
    success_count = 0
    for package in safe_packages:
        if run_command(f"pip install {package}", f"安装 {package}"):
            success_count += 1
    
    print(f"📊 安全依赖安装: {success_count}/{len(safe_packages)} 成功")
    return success_count >= len(safe_packages) - 2

def main():
    """主修复流程"""
    print("🔧 依赖冲突修复工具")
    print("=" * 50)
    print("🎯 解决NumPy版本冲突和torchao依赖问题")
    
    steps = [
        ("核弹级清理", nuclear_cleanup),
        ("安装兼容NumPy", install_numpy_compatible),
        ("安装稳定PyTorch", install_pytorch_stable),
        ("安装最小diffusers", install_diffusers_minimal),
        ("安装最小transformers", install_transformers_minimal),
        ("安装其他安全依赖", install_additional_safe),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"\n❌ {step_name}失败")
            print("🔧 建议:")
            print("   1. 重启Python内核")
            print("   2. 重新运行此脚本")
            print("   3. 或使用分阶段训练")
            return False
    
    print("\n🎉 依赖冲突修复完成！")
    print("✅ 现在可以运行:")
    print("   python test_api_compatibility.py")
    print("   python test_unified_environment.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
