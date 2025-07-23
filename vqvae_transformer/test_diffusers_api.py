#!/usr/bin/env python3
"""
专门测试diffusers API兼容性的脚本
检查diffusers 0.30.3版本的VQModel API是否发生变化
"""

import sys
import torch
import inspect
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_diffusers_vqmodel_api():
    """测试diffusers VQModel API"""
    print(f"🔍 测试diffusers VQModel API")
    print("=" * 60)
    
    try:
        import diffusers
        print(f"📦 diffusers版本: {diffusers.__version__}")
        
        # 检查VQModel导入路径
        try:
            from diffusers.models.autoencoders.vq_model import VQModel
            print(f"✅ VQModel导入成功 (路径: diffusers.models.autoencoders.vq_model)")
        except ImportError:
            try:
                from diffusers.models.vq_model import VQModel
                print(f"✅ VQModel导入成功 (路径: diffusers.models.vq_model)")
            except ImportError:
                try:
                    from diffusers import VQModel
                    print(f"✅ VQModel导入成功 (路径: diffusers)")
                except ImportError as e:
                    print(f"❌ VQModel导入失败: {e}")
                    return False
        
        # 检查VectorQuantizer导入路径
        try:
            from diffusers.models.autoencoders.vq_model import VectorQuantizer
            print(f"✅ VectorQuantizer导入成功 (路径: diffusers.models.autoencoders.vq_model)")
        except ImportError:
            try:
                from diffusers.models.vq_model import VectorQuantizer
                print(f"✅ VectorQuantizer导入成功 (路径: diffusers.models.vq_model)")
            except ImportError:
                try:
                    from diffusers import VectorQuantizer
                    print(f"✅ VectorQuantizer导入成功 (路径: diffusers)")
                except ImportError as e:
                    print(f"❌ VectorQuantizer导入失败: {e}")
                    return False
        
        # 检查VQModel构造函数参数
        print(f"\n🔍 检查VQModel构造函数参数...")
        sig = inspect.signature(VQModel.__init__)
        params = list(sig.parameters.keys())
        print(f"   参数数量: {len(params)}")
        print(f"   参数列表: {params}")
        
        # 检查关键参数是否存在
        required_params = ['in_channels', 'out_channels', 'latent_channels', 'num_vq_embeddings']
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            print(f"❌ 缺少关键参数: {missing_params}")
            return False
        else:
            print(f"✅ 所有关键参数都存在")
        
        # 测试VQModel实例化
        print(f"\n🧪 测试VQModel实例化...")
        try:
            model = VQModel(
                in_channels=3,
                out_channels=3,
                latent_channels=4,
                num_vq_embeddings=1024,
                vq_embed_dim=256,
            )
            print(f"✅ VQModel实例化成功")
        except Exception as e:
            print(f"❌ VQModel实例化失败: {e}")
            return False
        
        # 检查VQModel方法
        print(f"\n🔍 检查VQModel方法...")
        methods = [method for method in dir(model) if not method.startswith('_')]
        print(f"   方法数量: {len(methods)}")
        
        required_methods = ['encode', 'decode', 'forward']
        missing_methods = [m for m in required_methods if m not in methods]
        if missing_methods:
            print(f"❌ 缺少关键方法: {missing_methods}")
            return False
        else:
            print(f"✅ 所有关键方法都存在")
        
        # 测试前向传播
        print(f"\n🧪 测试VQModel前向传播...")
        x = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            # 测试encode
            try:
                encoded = model.encode(x)
                print(f"✅ encode方法成功")
                print(f"   输入形状: {x.shape}")
                if hasattr(encoded, 'latents'):
                    print(f"   编码输出形状: {encoded.latents.shape}")
                else:
                    print(f"   编码输出形状: {encoded.shape}")
            except Exception as e:
                print(f"❌ encode方法失败: {e}")
                return False
            
            # 测试decode
            try:
                if hasattr(encoded, 'latents'):
                    decoded = model.decode(encoded.latents)
                else:
                    decoded = model.decode(encoded)
                print(f"✅ decode方法成功")
                if hasattr(decoded, 'sample'):
                    print(f"   解码输出形状: {decoded.sample.shape}")
                else:
                    print(f"   解码输出形状: {decoded.shape}")
            except Exception as e:
                print(f"❌ decode方法失败: {e}")
                return False
            
            # 测试完整前向传播
            try:
                output = model(x)
                print(f"✅ 完整前向传播成功")
                if hasattr(output, 'sample'):
                    print(f"   输出形状: {output.sample.shape}")
                    print(f"   输出类型: {type(output)}")
                else:
                    print(f"   输出形状: {output.shape}")
                    print(f"   输出类型: {type(output)}")
            except Exception as e:
                print(f"❌ 完整前向传播失败: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_vqvae_inheritance():
    """测试自定义VQ-VAE的继承是否正确"""
    print(f"\n🔍 测试自定义VQ-VAE继承")
    print("=" * 60)
    
    try:
        from models.vqvae_model import MicroDopplerVQVAE
        
        # 检查继承关系
        from diffusers.models.autoencoders.vq_model import VQModel
        
        print(f"✅ MicroDopplerVQVAE导入成功")
        print(f"   是否继承自VQModel: {issubclass(MicroDopplerVQVAE, VQModel)}")
        
        # 测试实例化
        try:
            model = MicroDopplerVQVAE(
                in_channels=3,
                out_channels=3,
                latent_channels=4,
                num_vq_embeddings=1024,
                vq_embed_dim=256,
            )
            print(f"✅ MicroDopplerVQVAE实例化成功")
        except Exception as e:
            print(f"❌ MicroDopplerVQVAE实例化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试前向传播
        print(f"\n🧪 测试MicroDopplerVQVAE前向传播...")
        x = torch.randn(1, 3, 128, 128)
        
        with torch.no_grad():
            try:
                output = model(x)
                print(f"✅ 前向传播成功")
                print(f"   输入形状: {x.shape}")
                print(f"   输出形状: {output.sample.shape}")
                print(f"   输出类型: {type(output)}")
                
                # 检查输出属性
                if hasattr(output, 'vq_loss'):
                    print(f"   VQ损失: {output.vq_loss.item():.6f}")
                if hasattr(output, 'encoding_indices'):
                    print(f"   编码索引形状: {output.encoding_indices.shape}")
                
            except Exception as e:
                print(f"❌ 前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # 测试编码/解码
        print(f"\n🧪 测试编码/解码...")
        with torch.no_grad():
            try:
                # 编码
                encoded = model.encode(x, return_dict=True)
                print(f"✅ 编码成功")
                print(f"   latents形状: {encoded['latents'].shape}")
                print(f"   encoding_indices形状: {encoded['encoding_indices'].shape}")
                
                # 解码
                decoded = model.decode(encoded['latents'], force_not_quantize=True)
                print(f"✅ 解码成功")
                print(f"   解码输出形状: {decoded.shape}")
                
            except Exception as e:
                print(f"❌ 编码/解码失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print(f"🧪 diffusers API兼容性专项测试")
    print("=" * 80)
    
    # 测试基础diffusers VQModel
    diffusers_ok = test_diffusers_vqmodel_api()
    
    # 测试自定义VQ-VAE
    custom_ok = test_custom_vqvae_inheritance()
    
    # 总结
    print(f"\n📋 测试总结")
    print("=" * 60)
    
    if diffusers_ok and custom_ok:
        print(f"🎉 diffusers API兼容性测试通过！")
        print(f"   VQModel API正常工作")
        print(f"   自定义VQ-VAE继承正确")
    else:
        print(f"❌ 发现diffusers API问题：")
        if not diffusers_ok:
            print(f"   - diffusers VQModel API异常")
        if not custom_ok:
            print(f"   - 自定义VQ-VAE继承有问题")
        
        print(f"\n🔧 修复建议：")
        print(f"   1. 降级diffusers版本: pip install diffusers==0.21.0")
        print(f"   2. 检查VQModel API变化")
        print(f"   3. 更新自定义模型代码以适配新API")
    
    return diffusers_ok and custom_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
