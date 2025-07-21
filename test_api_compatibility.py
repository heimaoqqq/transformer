#!/usr/bin/env python3
"""
完整的API兼容性检查脚本
验证diffusers、transformers和自定义模型的API兼容性

⚠️ 重要：请在环境配置完成后运行此脚本
使用方法：
1. 先运行: python setup_unified_environment.py
2. 再运行: python test_api_compatibility.py
"""

import torch
import sys
import importlib
import inspect
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

class APICompatibilityChecker:
    """API兼容性检查器"""
    
    def __init__(self):
        self.results = {}
        self.warnings_captured = []
        
    def capture_warnings(self):
        """捕获警告信息"""
        def warning_handler(message, category, filename, lineno, file=None, line=None):
            self.warnings_captured.append({
                'message': str(message),
                'category': category.__name__,
                'filename': filename,
                'lineno': lineno
            })
        
        warnings.showwarning = warning_handler
        
    def check_module_versions(self) -> Dict[str, Any]:
        """检查模块版本"""
        print("🔍 检查模块版本...")
        
        modules = {
            'torch': 'PyTorch',
            'diffusers': 'Diffusers',
            'transformers': 'Transformers',
            'huggingface_hub': 'HuggingFace Hub',
            'accelerate': 'Accelerate',
            'safetensors': 'SafeTensors',
            'tokenizers': 'Tokenizers',
        }
        
        versions = {}
        for module_name, display_name in modules.items():
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, '__version__', 'unknown')
                versions[module_name] = version
                print(f"✅ {display_name}: {version}")
            except ImportError as e:
                versions[module_name] = f"ERROR: {e}"
                print(f"❌ {display_name}: 导入失败 - {e}")
        
        return versions
    
    def check_diffusers_api(self) -> Dict[str, Any]:
        """检查diffusers API兼容性"""
        print("\n🔍 检查diffusers API兼容性...")
        
        results = {}
        
        # 1. 检查VQModel导入
        try:
            from diffusers.models.autoencoders.vq_model import VQModel
            results['vqmodel_import'] = "SUCCESS"
            print("✅ VQModel导入成功")
            
            # 检查VQModel构造函数参数
            sig = inspect.signature(VQModel.__init__)
            params = list(sig.parameters.keys())
            results['vqmodel_params'] = params
            print(f"✅ VQModel参数: {len(params)} 个")
            
            # 测试VQModel实例化
            try:
                model = VQModel(
                    in_channels=3,
                    out_channels=3,
                    latent_channels=4,
                    num_vq_embeddings=1024,
                    vq_embed_dim=256,
                )
                results['vqmodel_instantiation'] = "SUCCESS"
                print("✅ VQModel实例化成功")
                
                # 检查VQModel方法
                methods = [method for method in dir(model) if not method.startswith('_')]
                results['vqmodel_methods'] = methods
                print(f"✅ VQModel方法: {len(methods)} 个")
                
            except Exception as e:
                results['vqmodel_instantiation'] = f"ERROR: {e}"
                print(f"❌ VQModel实例化失败: {e}")
                
        except ImportError as e:
            results['vqmodel_import'] = f"ERROR: {e}"
            print(f"❌ VQModel导入失败: {e}")
        
        # 2. 检查VectorQuantizer
        try:
            from diffusers.models.autoencoders.vq_model import VectorQuantizer
            results['vectorquantizer_import'] = "SUCCESS"
            print("✅ VectorQuantizer导入成功")
        except ImportError as e:
            results['vectorquantizer_import'] = f"ERROR: {e}"
            print(f"❌ VectorQuantizer导入失败: {e}")
        
        # 3. 检查其他diffusers组件
        try:
            from diffusers import AutoencoderKL
            results['autoencoder_import'] = "SUCCESS"
            print("✅ AutoencoderKL导入成功")
        except ImportError as e:
            results['autoencoder_import'] = f"ERROR: {e}"
            print(f"❌ AutoencoderKL导入失败: {e}")
        
        return results
    
    def check_transformers_api(self) -> Dict[str, Any]:
        """检查transformers API兼容性"""
        print("\n🔍 检查transformers API兼容性...")
        
        results = {}
        
        # 1. 检查GPT2
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
            results['gpt2_import'] = "SUCCESS"
            print("✅ GPT2导入成功")
            
            # 检查GPT2Config参数
            sig = inspect.signature(GPT2Config.__init__)
            params = list(sig.parameters.keys())
            results['gpt2_config_params'] = params
            print(f"✅ GPT2Config参数: {len(params)} 个")
            
            # 测试GPT2实例化
            try:
                config = GPT2Config(
                    vocab_size=1024,
                    n_positions=256,
                    n_embd=512,
                    n_layer=4,
                    n_head=8
                )
                model = GPT2LMHeadModel(config)
                results['gpt2_instantiation'] = "SUCCESS"
                print("✅ GPT2实例化成功")
                
            except Exception as e:
                results['gpt2_instantiation'] = f"ERROR: {e}"
                print(f"❌ GPT2实例化失败: {e}")
                
        except ImportError as e:
            results['gpt2_import'] = f"ERROR: {e}"
            print(f"❌ GPT2导入失败: {e}")
        
        # 2. 检查Tokenizer
        try:
            from transformers import AutoTokenizer
            results['tokenizer_import'] = "SUCCESS"
            print("✅ AutoTokenizer导入成功")
        except ImportError as e:
            results['tokenizer_import'] = f"ERROR: {e}"
            print(f"❌ AutoTokenizer导入失败: {e}")
        
        return results
    
    def check_custom_models_api(self) -> Dict[str, Any]:
        """检查自定义模型API兼容性"""
        print("\n🔍 检查自定义模型API兼容性...")
        
        results = {}
        
        try:
            from models.vqvae_model import MicroDopplerVQVAE
            results['custom_vqvae_import'] = "SUCCESS"
            print("✅ MicroDopplerVQVAE导入成功")
            
            # 检查构造函数参数
            sig = inspect.signature(MicroDopplerVQVAE.__init__)
            params = list(sig.parameters.keys())
            results['custom_vqvae_params'] = params
            print(f"✅ MicroDopplerVQVAE参数: {len(params)} 个")
            
            # 测试实例化
            try:
                model = MicroDopplerVQVAE(
                    in_channels=3,
                    out_channels=3,
                    latent_channels=4,
                    codebook_size=1024,
                    codebook_dim=256
                )
                results['custom_vqvae_instantiation'] = "SUCCESS"
                print("✅ MicroDopplerVQVAE实例化成功")
                
                # 检查方法
                methods = [method for method in dir(model) if not method.startswith('_')]
                results['custom_vqvae_methods'] = methods
                print(f"✅ MicroDopplerVQVAE方法: {len(methods)} 个")
                
            except Exception as e:
                results['custom_vqvae_instantiation'] = f"ERROR: {e}"
                print(f"❌ MicroDopplerVQVAE实例化失败: {e}")
                
        except ImportError as e:
            results['custom_vqvae_import'] = f"ERROR: {e}"
            print(f"❌ MicroDopplerVQVAE导入失败: {e}")
        
        return results
    
    def check_forward_compatibility(self) -> Dict[str, Any]:
        """检查前向传播兼容性"""
        print("\n🔍 检查前向传播兼容性...")
        
        results = {}
        
        # 1. 测试VQModel前向传播
        try:
            from diffusers.models.autoencoders.vq_model import VQModel
            model = VQModel(
                in_channels=3,
                out_channels=3,
                latent_channels=4,
                num_vq_embeddings=1024,
                vq_embed_dim=256,
            )
            
            x = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                output = model(x)
                
            results['vqmodel_forward'] = {
                'input_shape': list(x.shape),
                'output_shape': list(output.sample.shape),
                'output_type': type(output).__name__
            }
            print(f"✅ VQModel前向传播: {x.shape} -> {output.sample.shape}")
            
        except Exception as e:
            results['vqmodel_forward'] = f"ERROR: {e}"
            print(f"❌ VQModel前向传播失败: {e}")
        
        # 2. 测试GPT2前向传播
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
            config = GPT2Config(
                vocab_size=1024,
                n_positions=256,
                n_embd=512,
                n_layer=4,
                n_head=8
            )
            model = GPT2LMHeadModel(config)
            
            input_ids = torch.randint(0, 1024, (1, 10))
            with torch.no_grad():
                output = model(input_ids)
                
            results['gpt2_forward'] = {
                'input_shape': list(input_ids.shape),
                'output_shape': list(output.logits.shape),
                'output_type': type(output).__name__
            }
            print(f"✅ GPT2前向传播: {input_ids.shape} -> {output.logits.shape}")
            
        except Exception as e:
            results['gpt2_forward'] = f"ERROR: {e}"
            print(f"❌ GPT2前向传播失败: {e}")
        
        # 3. 测试自定义模型前向传播
        try:
            from models.vqvae_model import MicroDopplerVQVAE
            model = MicroDopplerVQVAE()
            
            x = torch.randn(1, 3, 128, 128)
            with torch.no_grad():
                output = model(x)
                
            results['custom_vqvae_forward'] = {
                'input_shape': list(x.shape),
                'output_shape': list(output.sample.shape),
                'output_type': type(output).__name__
            }
            print(f"✅ MicroDopplerVQVAE前向传播: {x.shape} -> {output.sample.shape}")
            
        except Exception as e:
            results['custom_vqvae_forward'] = f"ERROR: {e}"
            print(f"❌ MicroDopplerVQVAE前向传播失败: {e}")
        
        return results
    
    def check_save_load_compatibility(self) -> Dict[str, Any]:
        """检查保存/加载兼容性"""
        print("\n🔍 检查保存/加载兼容性...")
        
        results = {}
        
        try:
            from models.vqvae_model import MicroDopplerVQVAE
            model = MicroDopplerVQVAE()
            
            # 测试state_dict
            state_dict = model.state_dict()
            results['state_dict_keys'] = len(state_dict)
            print(f"✅ state_dict获取成功: {len(state_dict)} 个参数")
            
            # 测试load_state_dict
            model.load_state_dict(state_dict)
            results['load_state_dict'] = "SUCCESS"
            print("✅ load_state_dict成功")
            
        except Exception as e:
            results['save_load'] = f"ERROR: {e}"
            print(f"❌ 保存/加载测试失败: {e}")
        
        return results
    
    def run_full_check(self) -> Dict[str, Any]:
        """运行完整的API兼容性检查"""
        print("🎨 完整API兼容性检查")
        print("=" * 60)
        
        self.capture_warnings()
        
        # 运行所有检查
        self.results['versions'] = self.check_module_versions()
        self.results['diffusers_api'] = self.check_diffusers_api()
        self.results['transformers_api'] = self.check_transformers_api()
        self.results['custom_models_api'] = self.check_custom_models_api()
        self.results['forward_compatibility'] = self.check_forward_compatibility()
        self.results['save_load_compatibility'] = self.check_save_load_compatibility()
        self.results['warnings'] = self.warnings_captured
        
        return self.results
    
    def generate_report(self) -> str:
        """生成检查报告"""
        report = []
        report.append("# API兼容性检查报告")
        report.append("=" * 60)
        
        # 版本信息
        report.append("\n## 📦 版本信息")
        for module, version in self.results.get('versions', {}).items():
            status = "✅" if not version.startswith("ERROR") else "❌"
            report.append(f"{status} {module}: {version}")
        
        # 警告信息
        if self.warnings_captured:
            report.append(f"\n## ⚠️ 警告信息 ({len(self.warnings_captured)} 个)")
            for warning in self.warnings_captured:
                report.append(f"- {warning['category']}: {warning['message']}")
        
        # 总结
        total_checks = 0
        passed_checks = 0
        
        for category, results in self.results.items():
            if category in ['versions', 'warnings']:
                continue
            if isinstance(results, dict):
                for key, value in results.items():
                    total_checks += 1
                    if not str(value).startswith("ERROR"):
                        passed_checks += 1
        
        report.append(f"\n## 📊 检查总结")
        report.append(f"- 总检查项: {total_checks}")
        report.append(f"- 通过检查: {passed_checks}")
        report.append(f"- 成功率: {passed_checks/total_checks*100:.1f}%")
        
        if passed_checks >= total_checks * 0.9:  # 90%通过率
            report.append("\n🎉 API兼容性检查通过！环境配置正确。")
        else:
            report.append("\n❌ API兼容性检查失败，请检查环境配置。")
        
        return "\n".join(report)

def main():
    """主函数"""
    print("⚠️ 重要提醒：请确保已完成环境配置")
    print("   如果尚未配置环境，请先运行:")
    print("   python setup_unified_environment.py")
    print("   或")
    print("   python setup_vqvae_environment.py")
    print("   python setup_transformer_environment.py")
    print()

    checker = APICompatibilityChecker()
    results = checker.run_full_check()
    
    print("\n" + "=" * 60)
    print("📊 生成检查报告...")
    
    report = checker.generate_report()
    print(report)
    
    # 保存报告
    report_file = Path(__file__).parent / "api_compatibility_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 报告已保存到: {report_file}")
    
    return results

if __name__ == "__main__":
    results = main()
