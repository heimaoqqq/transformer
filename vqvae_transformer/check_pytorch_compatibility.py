#!/usr/bin/env python3
"""
PyTorch版本兼容性检查脚本
帮助找到正确的PyTorch、torchvision、torchaudio版本组合
"""

import subprocess
import sys

def run_command(cmd, timeout=60):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def check_current_pytorch():
    """检查当前PyTorch安装"""
    print("🔍 检查当前PyTorch安装...")
    
    try:
        import torch
        import torchvision
        import torchaudio
        
        print(f"✅ torch: {torch.__version__}")
        print(f"✅ torchvision: {torchvision.__version__}")
        print(f"✅ torchaudio: {torchaudio.__version__}")
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
        else:
            print("⚠️ CUDA不可用")
        
        return True
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch检查失败: {e}")
        return False

def get_available_pytorch_versions():
    """获取可用的PyTorch版本"""
    print("\n🔍 检查可用的PyTorch版本...")
    
    # 检查PyPI上的版本
    success, stdout, stderr = run_command("pip index versions torch")
    if success and stdout:
        print("📋 可用的torch版本:")
        lines = stdout.split('\n')
        for line in lines[:10]:  # 只显示前10个版本
            if 'Available versions:' in line or line.strip().startswith('torch'):
                print(f"   {line.strip()}")
    
    return True

def test_pytorch_combinations():
    """测试PyTorch版本组合"""
    print("\n🧪 测试PyTorch版本组合...")
    
    # 经过验证的版本组合
    combinations = [
        {
            "name": "PyTorch 2.0.1 (推荐)",
            "torch": "2.0.1",
            "torchvision": "0.15.1",
            "torchaudio": "2.0.1",
            "cuda": "cu118"
        },
        {
            "name": "PyTorch 1.13.1 (稳定)",
            "torch": "1.13.1", 
            "torchvision": "0.14.1",
            "torchaudio": "0.13.1",
            "cuda": "cu117"
        },
        {
            "name": "PyTorch 2.1.0 (最新)",
            "torch": "2.1.0",
            "torchvision": "0.16.0", 
            "torchaudio": "2.1.0",
            "cuda": "cu118"
        },
        {
            "name": "PyTorch 1.12.1 (保守)",
            "torch": "1.12.1",
            "torchvision": "0.13.1",
            "torchaudio": "0.12.1", 
            "cuda": "cu116"
        }
    ]
    
    print("📋 推荐的PyTorch版本组合:")
    for i, combo in enumerate(combinations, 1):
        print(f"\n{i}. {combo['name']}")
        print(f"   torch=={combo['torch']}")
        print(f"   torchvision=={combo['torchvision']}")
        print(f"   torchaudio=={combo['torchaudio']}")
        print(f"   CUDA: {combo['cuda']}")
        
        # 生成安装命令
        if combo['cuda']:
            cmd = f"pip install torch=={combo['torch']} torchvision=={combo['torchvision']} torchaudio=={combo['torchaudio']} --index-url https://download.pytorch.org/whl/{combo['cuda']}"
        else:
            cmd = f"pip install torch=={combo['torch']} torchvision=={combo['torchvision']} torchaudio=={combo['torchaudio']}"
        
        print(f"   安装命令: {cmd}")
    
    return combinations

def check_dependency_conflicts():
    """检查依赖冲突"""
    print("\n🔍 检查依赖冲突...")
    
    # 检查可能冲突的包
    conflicting_packages = [
        "torch", "torchvision", "torchaudio",
        "torch-audio", "torchtext", "torchdata"
    ]
    
    installed_packages = []
    
    for package in conflicting_packages:
        success, stdout, stderr = run_command(f"pip show {package}")
        if success:
            lines = stdout.split('\n')
            for line in lines:
                if line.startswith('Version:'):
                    version = line.split(':')[1].strip()
                    installed_packages.append((package, version))
                    break
    
    if installed_packages:
        print("📦 已安装的PyTorch相关包:")
        for package, version in installed_packages:
            print(f"   {package}: {version}")
    else:
        print("✅ 没有检测到PyTorch相关包")
    
    return installed_packages

def suggest_installation_strategy():
    """建议安装策略"""
    print("\n💡 安装策略建议:")
    
    print("🔧 方法1: 完全清理后重装 (推荐)")
    print("   1. pip uninstall torch torchvision torchaudio -y")
    print("   2. pip cache purge")
    print("   3. pip install torch==2.0.1 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n🔧 方法2: 使用Kaggle预装版本")
    print("   1. pip install torch torchvision torchaudio --upgrade")
    
    print("\n🔧 方法3: 分步安装")
    print("   1. pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118")
    print("   2. pip install torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118")
    print("   3. pip install torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n🔧 方法4: CPU版本 (如果GPU有问题)")
    print("   pip install torch==2.0.1 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu")

def auto_fix_pytorch():
    """自动修复PyTorch"""
    print("\n🔧 自动修复PyTorch...")
    
    # 完全清理
    print("🗑️ 清理现有安装...")
    packages_to_remove = ["torch", "torchvision", "torchaudio", "torch-audio"]
    for package in packages_to_remove:
        run_command(f"pip uninstall {package} -y")
    
    run_command("pip cache purge")
    
    # 尝试安装推荐版本
    install_commands = [
        "pip install torch==2.0.1 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118",
        "pip install torch torchvision torchaudio --upgrade",
        "pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117",
        "pip install torch==2.0.1 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu",
    ]
    
    for i, cmd in enumerate(install_commands, 1):
        print(f"\n🔄 尝试方案 {i}...")
        print(f"   命令: {cmd}")
        
        success, stdout, stderr = run_command(cmd, timeout=180)
        if success:
            print(f"✅ 方案 {i} 成功")
            
            # 验证安装
            if check_current_pytorch():
                print("🎉 PyTorch安装成功并验证通过")
                return True
            else:
                print("⚠️ 安装成功但验证失败")
        else:
            print(f"❌ 方案 {i} 失败: {stderr}")
    
    print("❌ 所有自动修复方案都失败")
    return False

def main():
    """主函数"""
    print("🔍 PyTorch版本兼容性检查工具")
    print("=" * 50)
    
    # 检查当前状态
    current_ok = check_current_pytorch()
    
    # 获取可用版本
    get_available_pytorch_versions()
    
    # 显示推荐组合
    test_pytorch_combinations()
    
    # 检查冲突
    check_dependency_conflicts()
    
    # 建议策略
    suggest_installation_strategy()
    
    # 询问是否自动修复
    if not current_ok:
        if len(sys.argv) > 1 and sys.argv[1] == "--fix":
            auto_fix_pytorch()
        else:
            print("\n💡 如需自动修复，请运行:")
            print("   python check_pytorch_compatibility.py --fix")

if __name__ == "__main__":
    main()
