#!/usr/bin/env python3
"""
专门修复 huggingface_hub 版本冲突的脚本
解决 'split_torch_state_dict_into_shards' 导入错误
"""

import subprocess
import sys

def run_command(cmd, description=""):
    """运行命令"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
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

def check_current_versions():
    """检查当前版本"""
    print("🔍 检查当前版本...")
    
    packages = ['huggingface_hub', 'diffusers', 'transformers']
    
    for package in packages:
        try:
            module = __import__(package.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            print(f"📦 {package}: {version}")
        except ImportError:
            print(f"❌ {package}: 未安装")
        except Exception as e:
            print(f"⚠️  {package}: 检查失败 - {e}")

def fix_huggingface_hub():
    """修复 huggingface_hub 版本问题"""
    print("\n🔧 修复 huggingface_hub 版本问题...")
    
    # 1. 卸载当前版本
    print("\n1️⃣ 卸载当前版本...")
    run_command("pip uninstall -y huggingface_hub", "卸载 huggingface_hub")
    
    # 2. 安装兼容版本
    print("\n2️⃣ 安装兼容版本...")
    compatible_versions = [
        "0.23.4",  # 较新但稳定
        "0.22.2",  # 稳定版本
        "0.21.4",  # 备选版本
        "0.20.3"   # 保守版本
    ]
    
    for version in compatible_versions:
        if run_command(f"pip install huggingface_hub=={version}", f"安装 huggingface_hub {version}"):
            print(f"✅ 成功安装 huggingface_hub {version}")
            break
    else:
        print("⚠️  所有指定版本都失败，尝试安装最新版本...")
        run_command("pip install huggingface_hub", "安装最新版 huggingface_hub")
    
    # 3. 重新安装 diffusers
    print("\n3️⃣ 重新安装 diffusers...")
    run_command("pip uninstall -y diffusers", "卸载 diffusers")
    run_command("pip install diffusers", "重新安装 diffusers")

def test_import():
    """测试导入"""
    print("\n🧪 测试导入...")
    
    # 清理模块缓存
    modules_to_clear = ['huggingface_hub', 'diffusers']
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
    
    try:
        # 测试 huggingface_hub
        import huggingface_hub
        print(f"✅ huggingface_hub: {huggingface_hub.__version__}")
        
        # 检查关键函数
        if hasattr(huggingface_hub, 'split_torch_state_dict_into_shards'):
            print("✅ split_torch_state_dict_into_shards 函数存在")
        else:
            print("⚠️  split_torch_state_dict_into_shards 函数不存在")
        
        # 测试 diffusers
        import diffusers
        print(f"✅ diffusers: {diffusers.__version__}")
        
        # 测试 AutoencoderKL 导入
        from diffusers import AutoencoderKL
        print("✅ AutoencoderKL 导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def test_vae_functionality():
    """测试VAE功能"""
    print("\n🔧 测试VAE功能...")
    
    try:
        from diffusers import AutoencoderKL
        import torch
        
        # 创建小模型
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            sample_size=32,
        )
        
        print("✅ VAE模型创建成功")
        
        # 测试前向传播
        with torch.no_grad():
            test_input = torch.randn(1, 3, 32, 32)
            latents = vae.encode(test_input).latent_dist.sample()
            reconstructed = vae.decode(latents).sample
            
            print(f"✅ VAE前向传播成功")
            print(f"   输入: {test_input.shape}")
            print(f"   潜在: {latents.shape}")
            print(f"   重建: {reconstructed.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ VAE功能测试失败: {e}")
        return False

def alternative_fix():
    """备选修复方案"""
    print("\n🔄 尝试备选修复方案...")
    
    # 方案1: 使用 --force-reinstall
    print("\n方案1: 强制重新安装...")
    run_command("pip install --force-reinstall huggingface_hub diffusers", "强制重新安装")
    
    # 方案2: 使用 --no-deps 避免依赖冲突
    print("\n方案2: 无依赖安装...")
    run_command("pip install --no-deps huggingface_hub==0.23.4", "无依赖安装 huggingface_hub")
    run_command("pip install diffusers", "重新安装 diffusers")

def main():
    """主函数"""
    print("🔧 HuggingFace Hub 修复工具")
    print("=" * 50)
    
    # 1. 检查当前版本
    check_current_versions()
    
    # 2. 修复版本问题
    fix_huggingface_hub()
    
    # 3. 测试导入
    print("\n" + "=" * 30 + " 测试阶段 " + "=" * 30)
    
    if test_import():
        print("✅ 导入测试通过")
        
        # 4. 测试VAE功能
        if test_vae_functionality():
            print("\n🎉 修复成功！VAE功能正常")
            print("\n📋 下一步:")
            print("   python train_kaggle.py --stage all")
        else:
            print("\n⚠️  VAE功能测试失败，尝试备选方案...")
            alternative_fix()
            
            # 再次测试
            if test_vae_functionality():
                print("✅ 备选方案成功！")
            else:
                print("❌ 备选方案也失败，可能需要重启内核")
    else:
        print("❌ 导入测试失败，尝试备选方案...")
        alternative_fix()

if __name__ == "__main__":
    main()
