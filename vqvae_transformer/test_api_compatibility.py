#!/usr/bin/env python3
"""
API兼容性测试脚本
测试不同版本diffusers的API路径和功能
"""

import importlib
import torch

def test_diffusers_version():
    """测试diffusers版本"""
    print("🔍 检查diffusers版本...")
    
    try:
        import diffusers
        version = diffusers.__version__
        print(f"✅ diffusers版本: {version}")
        return version
    except ImportError:
        print("❌ diffusers未安装")
        return None

def test_vqmodel_import_paths():
    """测试VQModel的不同导入路径"""
    print("\n🧪 测试VQModel导入路径...")
    
    import_paths = [
        ("diffusers.models.autoencoders.vq_model", "VQModel", "新版API (0.24.0+)"),
        ("diffusers.models.vq_model", "VQModel", "旧版API (0.20.0-0.23.x)"),
        ("diffusers", "VQModel", "直接导入 (更旧版本)"),
    ]
    
    successful_imports = []
    
    for module_path, class_name, description in import_paths:
        try:
            module = importlib.import_module(module_path)
            vq_class = getattr(module, class_name)
            print(f"✅ {description}: {module_path}.{class_name}")
            successful_imports.append((vq_class, description))
        except (ImportError, AttributeError) as e:
            print(f"❌ {description}: {e}")
    
    return successful_imports

def test_vqmodel_creation(vq_class, description):
    """测试VQModel创建"""
    print(f"\n🏗️ 测试VQModel创建 ({description})...")
    
    # 不同的配置参数组合
    configs = [
        {
            "name": "简单配置",
            "params": {
                "in_channels": 3,
                "out_channels": 3,
                "down_block_types": ["DownEncoderBlock2D"],
                "up_block_types": ["UpDecoderBlock2D"],
                "block_out_channels": [64],
                "layers_per_block": 1,
                "latent_channels": 64,
                "sample_size": 32,
                "num_vq_embeddings": 128,
                "norm_num_groups": 32,
                "vq_embed_dim": 64,
            }
        },
        {
            "name": "标准配置",
            "params": {
                "in_channels": 3,
                "out_channels": 3,
                "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
                "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
                "block_out_channels": [128, 256],
                "layers_per_block": 2,
                "latent_channels": 256,
                "sample_size": 64,
                "num_vq_embeddings": 512,
                "norm_num_groups": 32,
                "vq_embed_dim": 256,
            }
        }
    ]
    
    for config in configs:
        try:
            print(f"   尝试{config['name']}...")
            model = vq_class(**config['params'])
            print(f"   ✅ {config['name']}创建成功")
            
            # 测试前向传播
            sample_size = config['params']['sample_size']
            test_input = torch.randn(1, 3, sample_size, sample_size)
            
            with torch.no_grad():
                result = model.encode(test_input)
                print(f"   ✅ 编码成功: {result.latents.shape}")
                
                decoded = model.decode(result.latents)
                print(f"   ✅ 解码成功: {decoded.sample.shape}")
            
            return True, config['name']
            
        except Exception as e:
            print(f"   ❌ {config['name']}失败: {e}")
            continue
    
    return False, None

def test_huggingface_hub():
    """测试HuggingFace Hub API"""
    print("\n🤗 测试HuggingFace Hub API...")
    
    try:
        import huggingface_hub
        version = huggingface_hub.__version__
        print(f"✅ huggingface_hub版本: {version}")
        
        # 测试cached_download
        try:
            from huggingface_hub import cached_download
            print("✅ cached_download: 可用")
            return True
        except ImportError as e:
            print(f"❌ cached_download: 不可用 - {e}")
            
            # 尝试新的API
            try:
                from huggingface_hub import hf_hub_download
                print("✅ hf_hub_download: 可用 (新API)")
                return True
            except ImportError:
                print("❌ 所有下载API都不可用")
                return False
                
    except ImportError:
        print("❌ huggingface_hub未安装")
        return False

def test_transformers_api():
    """测试Transformers API"""
    print("\n🤖 测试Transformers API...")
    
    try:
        import transformers
        version = transformers.__version__
        print(f"✅ transformers版本: {version}")
        
        # 测试GPT2
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
        print("✅ GPT2模型创建成功")
        
        # 测试前向传播
        test_input = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ GPT2前向传播成功: {output.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformers测试失败: {e}")
        return False

def generate_compatibility_report():
    """生成兼容性报告"""
    print("\n📄 生成兼容性报告...")
    
    report = "# VQ-VAE + Transformer API兼容性报告\n\n"
    
    # 版本信息
    report += "## 版本信息\n"
    
    packages = ['torch', 'diffusers', 'transformers', 'huggingface_hub', 'accelerate']
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            report += f"- {package}: {version}\n"
        except ImportError:
            report += f"- {package}: 未安装\n"
    
    report += "\n## 测试结果\n"
    
    # 运行测试
    diffusers_version = test_diffusers_version()
    vqmodel_imports = test_vqmodel_import_paths()
    hub_ok = test_huggingface_hub()
    transformers_ok = test_transformers_api()
    
    # VQModel测试
    vqmodel_ok = False
    working_config = None
    
    for vq_class, description in vqmodel_imports:
        success, config_name = test_vqmodel_creation(vq_class, description)
        if success:
            vqmodel_ok = True
            working_config = f"{description} - {config_name}"
            break
    
    # 添加到报告
    report += f"- diffusers版本: {diffusers_version or '未安装'}\n"
    report += f"- VQModel导入: {'✅ 成功' if vqmodel_imports else '❌ 失败'}\n"
    report += f"- VQModel创建: {'✅ 成功' if vqmodel_ok else '❌ 失败'}\n"
    if working_config:
        report += f"  - 工作配置: {working_config}\n"
    report += f"- HuggingFace Hub: {'✅ 成功' if hub_ok else '❌ 失败'}\n"
    report += f"- Transformers: {'✅ 成功' if transformers_ok else '❌ 失败'}\n"
    
    # 总体评估
    report += "\n## 总体评估\n"
    
    if vqmodel_ok and hub_ok and transformers_ok:
        report += "✅ **环境完全兼容** - 可以开始训练\n"
    elif vqmodel_ok and transformers_ok:
        report += "⚠️ **基本兼容** - 可以训练，但可能有下载问题\n"
    else:
        report += "❌ **环境不兼容** - 需要修复环境\n"
        report += "\n### 建议修复步骤:\n"
        if not vqmodel_ok:
            report += "1. 运行: `python setup_environment.py`\n"
        if not hub_ok:
            report += "2. 安装兼容的huggingface_hub: `pip install huggingface-hub==0.17.3`\n"
        if not transformers_ok:
            report += "3. 安装兼容的transformers: `pip install transformers==4.35.2`\n"
    
    # 保存报告
    with open("api_compatibility_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ 兼容性报告保存到: api_compatibility_report.md")
    
    return vqmodel_ok and hub_ok and transformers_ok

def main():
    """主函数"""
    print("🔬 VQ-VAE + Transformer API兼容性测试")
    print("=" * 60)
    
    # 运行所有测试并生成报告
    overall_success = generate_compatibility_report()
    
    print(f"\n{'='*20} 测试总结 {'='*20}")
    
    if overall_success:
        print("🎉 所有API测试通过!")
        print("✅ 环境完全兼容，可以开始训练")
        print("\n🚀 下一步:")
        print("   python train_main.py --data_dir /path/to/data")
    else:
        print("⚠️ 部分API测试失败")
        print("💡 建议:")
        print("1. 查看详细报告: api_compatibility_report.md")
        print("2. 运行修复脚本: python quick_fix_environment.py")
        print("3. 重新测试: python test_api_compatibility.py")

if __name__ == "__main__":
    main()
