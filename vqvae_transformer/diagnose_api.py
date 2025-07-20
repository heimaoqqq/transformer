#!/usr/bin/env python3
"""
HuggingFace API兼容性诊断脚本
专门诊断cached_download等API问题
"""

import sys
import importlib
import subprocess

def check_package_version(package_name):
    """检查包版本"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"📦 {package_name}: {version}")
        return version
    except ImportError:
        print(f"❌ {package_name}: 未安装")
        return None

def test_huggingface_apis():
    """测试HuggingFace各种API"""
    print("\n🧪 测试HuggingFace API...")
    
    # 检查版本
    hf_version = check_package_version("huggingface_hub")
    if not hf_version:
        return False
    
    # 测试各种API
    apis_to_test = [
        ("cached_download", "from huggingface_hub import cached_download"),
        ("hf_hub_download", "from huggingface_hub import hf_hub_download"),
        ("snapshot_download", "from huggingface_hub import snapshot_download"),
        ("login", "from huggingface_hub import login"),
        ("HfApi", "from huggingface_hub import HfApi"),
    ]
    
    available_apis = []
    
    for api_name, import_cmd in apis_to_test:
        try:
            exec(import_cmd)
            available_apis.append(api_name)
            print(f"✅ {api_name}: 可用")
        except ImportError as e:
            print(f"❌ {api_name}: 不可用 - {e}")
        except Exception as e:
            print(f"⚠️ {api_name}: 其他问题 - {e}")
    
    return available_apis

def test_diffusers_apis():
    """测试Diffusers API"""
    print("\n🎨 测试Diffusers API...")
    
    # 检查版本
    diffusers_version = check_package_version("diffusers")
    if not diffusers_version:
        return False
    
    # 测试VQModel导入路径
    vqmodel_paths = [
        ("diffusers.models.autoencoders.vq_model", "VQModel", "新版API (0.24.0+)"),
        ("diffusers.models.vq_model", "VQModel", "旧版API (0.20.0-0.23.x)"),
        ("diffusers", "VQModel", "直接导入"),
    ]
    
    available_paths = []
    
    for module_path, class_name, description in vqmodel_paths:
        try:
            if module_path == "diffusers":
                exec(f"from {module_path} import {class_name}")
            else:
                module = importlib.import_module(module_path)
                getattr(module, class_name)
            
            available_paths.append((module_path, description))
            print(f"✅ {description}: {module_path}.{class_name}")
        except (ImportError, AttributeError) as e:
            print(f"❌ {description}: {e}")
    
    return available_paths

def check_dependencies():
    """检查依赖包"""
    print("\n📋 检查依赖包...")
    
    dependencies = [
        "torch", "transformers", "accelerate", "tokenizers", 
        "safetensors", "numpy", "pillow", "requests"
    ]
    
    for dep in dependencies:
        check_package_version(dep)

def suggest_fixes(available_hf_apis, available_vq_paths):
    """建议修复方案"""
    print("\n💡 修复建议...")
    
    if not available_hf_apis:
        print("🔧 HuggingFace API问题:")
        print("1. 尝试重装: pip install huggingface_hub==0.17.3 --force-reinstall --no-cache-dir")
        print("2. 尝试降级: pip install huggingface_hub==0.16.4")
        print("3. 清理缓存: pip cache purge")
        print("4. 重启Python内核")
    
    elif "cached_download" not in available_hf_apis:
        print("🔧 cached_download不可用:")
        if "hf_hub_download" in available_hf_apis:
            print("✅ 可以使用 hf_hub_download 替代")
            print("   示例: hf_hub_download(repo_id='repo', filename='file.bin')")
        elif "snapshot_download" in available_hf_apis:
            print("✅ 可以使用 snapshot_download 替代")
            print("   示例: snapshot_download(repo_id='repo')")
        else:
            print("❌ 需要重装huggingface_hub")
    
    if not available_vq_paths:
        print("🔧 VQModel API问题:")
        print("1. 尝试重装: pip install diffusers==0.24.0 --force-reinstall")
        print("2. 检查diffusers版本兼容性")
        print("3. 重启Python内核")
    
    else:
        working_path = available_vq_paths[0]
        print(f"✅ VQModel可用路径: {working_path[0]}")

def run_quick_fix():
    """运行快速修复"""
    print("\n🔧 执行快速修复...")
    
    commands = [
        ("pip uninstall huggingface_hub diffusers -y", "卸载问题包"),
        ("pip cache purge", "清理缓存"),
        ("pip install huggingface_hub==0.17.3 --no-cache-dir", "重装huggingface_hub"),
        ("pip install diffusers==0.24.0 --no-cache-dir", "重装diffusers"),
    ]
    
    for cmd, desc in commands:
        print(f"🔄 {desc}...")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"✅ {desc} 成功")
            else:
                print(f"❌ {desc} 失败: {result.stderr.strip()}")
        except Exception as e:
            print(f"❌ {desc} 异常: {e}")

def main():
    """主函数"""
    print("🔍 HuggingFace API兼容性诊断工具")
    print("=" * 50)
    
    # 检查依赖
    check_dependencies()
    
    # 测试API
    available_hf_apis = test_huggingface_apis()
    available_vq_paths = test_diffusers_apis()
    
    # 生成建议
    suggest_fixes(available_hf_apis, available_vq_paths)
    
    # 询问是否执行快速修复
    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        run_quick_fix()
        print("\n🔄 重新测试...")
        test_huggingface_apis()
        test_diffusers_apis()
    else:
        print("\n💡 如需自动修复，请运行:")
        print("   python diagnose_api.py --fix")

if __name__ == "__main__":
    main()
