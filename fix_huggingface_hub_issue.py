#!/usr/bin/env python3
"""
修复 huggingface_hub cached_download 兼容性问题
恢复到稳定的版本组合
"""

import subprocess
import sys

def run_command(cmd, description=""):
    """运行命令"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - 完成")
            return True
        else:
            print(f"❌ {description} - 失败")
            if result.stderr:
                print(f"错误: {result.stderr[:200]}...")
            return False
    except Exception as e:
        print(f"❌ {description} - 异常: {e}")
        return False

def check_current_issue():
    """检查当前问题"""
    print("🔍 检查当前 cached_download 问题:")
    
    try:
        from huggingface_hub import cached_download
        print("   ✅ cached_download 可用")
        return False  # 没有问题
    except ImportError as e:
        print(f"   ❌ cached_download 不可用: {e}")
        return True  # 有问题
    except Exception as e:
        print(f"   ❌ 其他错误: {e}")
        return True

def check_versions():
    """检查当前版本"""
    print("\n📦 检查当前版本:")
    
    packages = ['huggingface_hub', 'diffusers', 'transformers', 'accelerate']
    versions = {}
    
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"   {package}: {version}")
            versions[package] = version
        except ImportError:
            print(f"   {package}: 未安装")
            versions[package] = None
    
    return versions

def fix_version_compatibility():
    """修复版本兼容性"""
    print("\n🔧 修复版本兼容性问题:")
    
    # 稳定的版本组合 (经过验证)
    stable_versions = [
        "huggingface_hub==0.16.4",  # 包含 cached_download
        "diffusers==0.21.4",        # 与 huggingface_hub 0.16.4 兼容
        "transformers==4.30.2",     # 稳定版本
        "accelerate==0.20.3"        # 稳定版本
    ]
    
    print("   安装稳定版本组合...")
    
    success = True
    for package in stable_versions:
        if not run_command(f"pip install {package}", f"安装 {package}"):
            success = False
    
    return success

def verify_fix():
    """验证修复结果"""
    print("\n✅ 验证修复结果:")
    
    try:
        # 测试 cached_download
        from huggingface_hub import cached_download
        print("   ✅ cached_download 导入成功")
        
        # 测试 diffusers
        from diffusers import AutoencoderKL, UNet2DConditionModel
        print("   ✅ diffusers 导入成功")
        
        # 测试创建模型
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            sample_size=128,
        )
        print("   ✅ VAE 创建成功")
        
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            cross_attention_dim=768,
        )
        print("   ✅ UNet 创建成功")
        
        print("   🎉 所有组件工作正常！")
        return True
        
    except Exception as e:
        print(f"   ❌ 验证失败: {e}")
        return False

def show_solution_summary():
    """显示解决方案总结"""
    print("\n📋 解决方案总结:")
    print("=" * 50)
    
    print("\n🔍 问题原因:")
    print("   - huggingface_hub 新版本移除了 cached_download 函数")
    print("   - diffusers 仍然依赖 cached_download")
    print("   - 版本不兼容导致导入失败")
    
    print("\n🔧 解决方案:")
    print("   - 恢复到稳定的版本组合:")
    print("     * huggingface_hub==0.16.4 (包含 cached_download)")
    print("     * diffusers==0.21.4 (兼容旧版 huggingface_hub)")
    print("     * transformers==4.30.2")
    print("     * accelerate==0.20.3")
    
    print("\n✅ 验证方法:")
    print("   python fix_huggingface_hub_issue.py")
    print("   python verify_api_compatibility.py")

def main():
    """主函数"""
    print("🔧 修复 huggingface_hub cached_download 兼容性问题")
    print("=" * 60)
    
    # 1. 检查当前问题
    has_issue = check_current_issue()
    
    # 2. 检查当前版本
    versions = check_versions()
    
    if has_issue:
        print("\n⚠️  确认存在 cached_download 问题")
        
        # 3. 修复版本兼容性
        if fix_version_compatibility():
            print("\n🔄 重新验证...")
            
            # 4. 验证修复
            if verify_fix():
                print("\n🎉 问题修复成功！")
                show_solution_summary()
                return True
            else:
                print("\n❌ 修复验证失败")
                return False
        else:
            print("\n❌ 版本修复失败")
            return False
    else:
        print("\n✅ 没有发现 cached_download 问题")
        return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ 可以继续进行 VAE 和 LDM 训练")
    else:
        print("\n❌ 需要手动修复版本问题")
        print("\n🔧 手动修复命令:")
        print("pip install huggingface_hub==0.16.4 diffusers==0.21.4 transformers==4.30.2 accelerate==0.20.3")
