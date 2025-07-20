#!/usr/bin/env python3
"""
HuggingFace API迁移指南
从cached_download迁移到hf_hub_download
"""

def show_api_migration():
    """显示API迁移指南"""
    print("🔄 HuggingFace API迁移指南")
    print("=" * 50)
    print("📋 从 cached_download 迁移到 hf_hub_download")
    
    print("\n❌ 旧API (cached_download):")
    print("""
from huggingface_hub import cached_download

# 旧的用法
model_path = cached_download(
    url="https://huggingface.co/repo/resolve/main/model.bin",
    cache_dir="/path/to/cache"
)
""")
    
    print("✅ 新API (hf_hub_download):")
    print("""
from huggingface_hub import hf_hub_download

# 新的用法
model_path = hf_hub_download(
    repo_id="repo",
    filename="model.bin",
    cache_dir="/path/to/cache"
)
""")
    
    print("\n🔧 常见迁移场景:")
    
    scenarios = [
        {
            "name": "下载模型文件",
            "old": """cached_download("https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.bin")""",
            "new": """hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename="vae/diffusion_pytorch_model.bin")"""
        },
        {
            "name": "下载配置文件",
            "old": """cached_download("https://huggingface.co/repo/resolve/main/config.json")""",
            "new": """hf_hub_download(repo_id="repo", filename="config.json")"""
        },
        {
            "name": "指定缓存目录",
            "old": """cached_download(url, cache_dir="/custom/cache")""",
            "new": """hf_hub_download(repo_id="repo", filename="file", cache_dir="/custom/cache")"""
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}:")
        print(f"   ❌ 旧: {scenario['old']}")
        print(f"   ✅ 新: {scenario['new']}")

def create_compatibility_wrapper():
    """创建兼容性包装器"""
    print("\n🛠️ 兼容性包装器代码:")
    print("""
def safe_download(url=None, repo_id=None, filename=None, cache_dir=None):
    \"\"\"
    兼容性下载函数，自动选择可用的API
    \"\"\"
    try:
        # 尝试使用新API
        from huggingface_hub import hf_hub_download
        
        if repo_id and filename:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir
            )
        elif url:
            # 从URL解析repo_id和filename
            import re
            match = re.match(r'https://huggingface.co/([^/]+/[^/]+)/resolve/main/(.+)', url)
            if match:
                repo_id, filename = match.groups()
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=cache_dir
                )
        
        raise ValueError("无法解析下载参数")
        
    except ImportError:
        # 回退到旧API
        from huggingface_hub import cached_download
        return cached_download(url, cache_dir=cache_dir)

# 使用示例
model_path = safe_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    filename="vae/diffusion_pytorch_model.bin"
)
""")

def test_available_apis():
    """测试可用的API"""
    print("\n🧪 测试可用的下载API:")
    
    apis = [
        ("cached_download", "from huggingface_hub import cached_download"),
        ("hf_hub_download", "from huggingface_hub import hf_hub_download"),
        ("snapshot_download", "from huggingface_hub import snapshot_download"),
    ]
    
    available = []
    
    for api_name, import_cmd in apis:
        try:
            exec(import_cmd)
            print(f"✅ {api_name}: 可用")
            available.append(api_name)
        except ImportError as e:
            print(f"❌ {api_name}: 不可用 - {e}")
    
    if available:
        print(f"\n📊 可用API: {', '.join(available)}")
        
        if "hf_hub_download" in available:
            print("💡 推荐使用 hf_hub_download (新API)")
        elif "cached_download" in available:
            print("💡 可以使用 cached_download (旧API)")
        else:
            print("⚠️ 只有 snapshot_download 可用")
    else:
        print("❌ 没有可用的下载API")
    
    return available

def check_huggingface_hub_version():
    """检查huggingface_hub版本"""
    print("\n📦 检查huggingface_hub版本:")
    
    try:
        import huggingface_hub
        version = huggingface_hub.__version__
        print(f"✅ huggingface_hub版本: {version}")
        
        # 版本兼容性说明
        version_info = [
            ("< 0.14.0", "支持 cached_download"),
            ("0.14.0 - 0.19.3", "cached_download 逐步废弃"),
            (">= 0.19.4", "推荐使用 hf_hub_download"),
            (">= 0.20.0", "cached_download 可能完全移除"),
        ]
        
        print("\n📋 版本兼容性:")
        for version_range, description in version_info:
            print(f"   {version_range}: {description}")
        
        # 解析当前版本
        try:
            from packaging import version as pkg_version
            current_version = pkg_version.parse(version)
            
            if current_version >= pkg_version.parse("0.19.4"):
                print(f"\n💡 当前版本 {version} 建议使用 hf_hub_download")
            else:
                print(f"\n💡 当前版本 {version} 可以使用 cached_download")
                
        except ImportError:
            print("\n⚠️ 无法解析版本，请手动检查API兼容性")
        
        return version
        
    except ImportError:
        print("❌ huggingface_hub 未安装")
        return None

def main():
    """主函数"""
    print("🔄 HuggingFace API迁移助手")
    print("=" * 60)
    
    # 检查版本
    version = check_huggingface_hub_version()
    
    # 测试API
    available_apis = test_available_apis()
    
    # 显示迁移指南
    show_api_migration()
    
    # 显示兼容性包装器
    create_compatibility_wrapper()
    
    # 总结建议
    print(f"\n{'='*20} 总结建议 {'='*20}")
    
    if "hf_hub_download" in available_apis:
        print("✅ 推荐使用 hf_hub_download (新API)")
        print("   - 更稳定，官方推荐")
        print("   - 参数更清晰")
        print("   - 长期支持")
    elif "cached_download" in available_apis:
        print("⚠️ 可以使用 cached_download (旧API)")
        print("   - 可能在未来版本中移除")
        print("   - 建议尽快迁移到新API")
    else:
        print("❌ 需要安装兼容版本的 huggingface_hub")
        print("   建议: pip install 'huggingface_hub>=0.19.4'")

if __name__ == "__main__":
    main()
