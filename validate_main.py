#!/usr/bin/env python3
"""
VQ-VAE + Transformer 主验证脚本
验证生成图像的质量和用户特征保持度
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# 条件导入模型 - 检查环境兼容性
try:
    from models.vqvae_model import MicroDopplerVQVAE
    VQVAE_AVAILABLE = True
except ImportError as e:
    print(f"❌ 无法导入VQ-VAE模型: {e}")
    print("   请确保在正确的环境中运行验证脚本")
    VQVAE_AVAILABLE = False

try:
    from models.transformer_model import MicroDopplerTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError as e:
    print(f"❌ 无法导入Transformer模型: {e}")
    print("   请确保在正确的环境中运行验证脚本")
    TRANSFORMER_AVAILABLE = False

# 检查必要组件
if not (VQVAE_AVAILABLE and TRANSFORMER_AVAILABLE):
    print("❌ 验证脚本需要同时支持VQ-VAE和Transformer")
    print("   建议在Transformer环境中运行，因为它可以加载VQ-VAE模型")
    sys.exit(1)

# 尝试导入主项目的验证器
try:
    from validation.metric_learning_validator import MetricLearningValidator
except ImportError:
    print("⚠️ 无法导入主项目验证器，将使用简化验证")
    MetricLearningValidator = None

class VQVAEValidator:
    """VQ-VAE + Transformer 专用验证器"""
    
    def __init__(self, model_dir, device="auto"):
        self.model_dir = Path(model_dir)
        
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🎮 使用设备: {self.device}")
        
        # 初始化度量学习验证器
        if MetricLearningValidator:
            self.metric_validator = MetricLearningValidator(device)
        else:
            self.metric_validator = None
        
        # 加载模型用于分析
        try:
            self.vqvae_model = self._load_vqvae()
            self.transformer_model = self._load_transformer()
            print(f"✅ 模型加载完成")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            self.vqvae_model = None
            self.transformer_model = None
    
    def _load_vqvae(self):
        """加载VQ-VAE模型"""
        vqvae_path = self.model_dir / "vqvae"
        
        checkpoint_path = vqvae_path / "best_model.pth"
        if not checkpoint_path.exists():
            checkpoint_path = vqvae_path / "final_model.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"未找到VQ-VAE模型: {vqvae_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        model = MicroDopplerVQVAE(
            num_vq_embeddings=checkpoint['args'].codebook_size,
            commitment_cost=checkpoint['args'].commitment_cost,
            ema_decay=checkpoint['args'].ema_decay,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def _load_transformer(self):
        """加载Transformer模型"""
        transformer_path = self.model_dir / "transformer"
        
        checkpoint_path = transformer_path / "best_model.pth"
        if not checkpoint_path.exists():
            checkpoint_path = transformer_path / "final_model.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"未找到Transformer模型: {transformer_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        args = checkpoint['args']
        
        model = MicroDopplerTransformer(
            vocab_size=args.codebook_size,
            max_seq_len=getattr(args, 'max_seq_len', 256),
            num_users=args.num_users,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            dropout=args.dropout,
            use_cross_attention=args.use_cross_attention,
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def validate_comprehensive(
        self,
        real_data_dir: str,
        generated_data_dir: str,
        target_user_id: int,
        threshold: float = 0.3,
    ):
        """
        综合验证
        Args:
            real_data_dir: 真实数据目录
            generated_data_dir: 生成数据目录
            target_user_id: 目标用户ID
            threshold: 相似性阈值
        """
        print(f"\n🔍 VQ-VAE + Transformer 综合验证")
        print("=" * 60)
        
        results = {}
        
        # 1. 码本使用情况分析
        if self.vqvae_model is not None:
            print(f"\n📊 码本使用情况分析:")
            codebook_stats = self.vqvae_model.get_codebook_stats()
            results['codebook_stats'] = codebook_stats
            
            print(f"   活跃码本: {codebook_stats['active_codes']}/{codebook_stats['total_codes']}")
            print(f"   使用率: {codebook_stats['usage_rate']:.3f}")
            print(f"   使用熵: {codebook_stats['usage_entropy']:.3f}")
            
            # 可视化码本使用
            usage_plot_path = Path(generated_data_dir).parent / "codebook_usage_validation.png"
            self.vqvae_model.plot_codebook_usage(str(usage_plot_path))
            print(f"   📈 码本使用图保存到: {usage_plot_path}")
        
        # 2. 度量学习验证
        if self.metric_validator:
            print(f"\n🧠 度量学习验证:")
            
            # 加载真实数据
            user_images = self.metric_validator.load_user_images(real_data_dir)
            if not user_images:
                print("❌ 未找到真实数据")
                return results
            
            # 训练Siamese网络
            history = self.metric_validator.train_siamese_network(
                user_images,
                epochs=30,  # 适中的训练轮数
                batch_size=16,
            )
            results['training_history'] = history
            
            # 计算用户原型
            self.metric_validator.compute_user_prototypes(user_images)
            
            # 验证生成图像
            validation_result = self.metric_validator.validate_generated_images(
                target_user_id=target_user_id,
                generated_images_dir=generated_data_dir,
                threshold=threshold,
            )
            results['validation_result'] = validation_result
        else:
            print(f"\n⚠️ 跳过度量学习验证 (验证器不可用)")
        
        # 3. 多样性分析
        print(f"\n🎨 多样性分析:")
        diversity_stats = self._analyze_diversity(generated_data_dir, target_user_id)
        results['diversity_stats'] = diversity_stats
        
        # 4. 生成质量评估
        print(f"\n⭐ 质量评估:")
        quality_stats = self._analyze_quality(
            results.get('validation_result'), 
            results.get('codebook_stats')
        )
        results['quality_stats'] = quality_stats
        
        # 5. 保存验证报告
        self._save_validation_report(results, generated_data_dir)
        
        return results
    
    def _analyze_diversity(self, generated_data_dir, target_user_id):
        """分析生成图像的多样性"""
        generated_path = Path(generated_data_dir)
        
        # 查找目标用户的生成图像
        user_dir = generated_path / f"user_{target_user_id:02d}"
        if not user_dir.exists():
            # 如果没有用户子目录，直接在根目录查找
            image_files = list(generated_path.glob("*.png")) + list(generated_path.glob("*.jpg"))
        else:
            image_files = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))
        
        num_images = len(image_files)
        
        diversity_stats = {
            'num_generated_images': num_images,
            'diversity_score': 0.0,
            'uniqueness_ratio': 0.0,
        }
        
        if num_images > 1:
            # 简单的多样性评估：基于文件大小的变异系数
            file_sizes = [f.stat().st_size for f in image_files]
            if len(file_sizes) > 1:
                size_std = np.std(file_sizes)
                size_mean = np.mean(file_sizes)
                diversity_score = size_std / size_mean if size_mean > 0 else 0
                diversity_stats['diversity_score'] = diversity_score
                diversity_stats['uniqueness_ratio'] = min(1.0, diversity_score * 10)  # 归一化
        
        print(f"   生成图像数量: {num_images}")
        print(f"   多样性分数: {diversity_stats['diversity_score']:.3f}")
        print(f"   独特性比率: {diversity_stats['uniqueness_ratio']:.3f}")
        
        return diversity_stats
    
    def _analyze_quality(self, validation_result, codebook_stats):
        """分析整体质量"""
        quality_stats = {
            'overall_score': 0.0,
            'user_fidelity': 0.0,
            'generation_diversity': 0.0,
            'model_efficiency': 0.0,
        }
        
        if validation_result:
            # 用户保真度 (基于相似性)
            user_fidelity = validation_result.get('avg_similarity', 0)
            quality_stats['user_fidelity'] = user_fidelity
            
            # 成功率作为质量指标
            success_rate = validation_result.get('success_rate', 0)
            
            print(f"   用户保真度: {user_fidelity:.3f}")
            print(f"   验证成功率: {success_rate:.3f}")
        
        if codebook_stats:
            # 模型效率 (基于码本使用率)
            model_efficiency = codebook_stats.get('usage_rate', 0)
            quality_stats['model_efficiency'] = model_efficiency
            
            print(f"   模型效率: {model_efficiency:.3f}")
        
        # 计算综合分数
        overall_score = (
            quality_stats['user_fidelity'] * 0.4 +
            quality_stats['generation_diversity'] * 0.3 +
            quality_stats['model_efficiency'] * 0.3
        )
        quality_stats['overall_score'] = overall_score
        
        print(f"   综合质量分数: {overall_score:.3f}")
        
        return quality_stats
    
    def _save_validation_report(self, results, output_dir):
        """保存验证报告"""
        report_path = Path(output_dir).parent / "validation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("VQ-VAE + Transformer 验证报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 码本统计
            if 'codebook_stats' in results:
                stats = results['codebook_stats']
                f.write("📊 码本使用情况:\n")
                f.write(f"   活跃码本: {stats['active_codes']}/{stats['total_codes']}\n")
                f.write(f"   使用率: {stats['usage_rate']:.3f}\n")
                f.write(f"   使用熵: {stats['usage_entropy']:.3f}\n\n")
            
            # 验证结果
            if 'validation_result' in results:
                result = results['validation_result']
                f.write("🧠 度量学习验证:\n")
                f.write(f"   成功率: {result.get('success_rate', 'N/A'):.3f}\n")
                f.write(f"   平均相似性: {result.get('avg_similarity', 'N/A'):.3f}\n")
                f.write(f"   成功图像: {result.get('successful_images', 'N/A')}/{result.get('total_images', 'N/A')}\n\n")
            
            # 多样性分析
            if 'diversity_stats' in results:
                stats = results['diversity_stats']
                f.write("🎨 多样性分析:\n")
                f.write(f"   生成图像数量: {stats['num_generated_images']}\n")
                f.write(f"   多样性分数: {stats['diversity_score']:.3f}\n")
                f.write(f"   独特性比率: {stats['uniqueness_ratio']:.3f}\n\n")
            
            # 质量评估
            if 'quality_stats' in results:
                stats = results['quality_stats']
                f.write("⭐ 质量评估:\n")
                f.write(f"   综合质量分数: {stats['overall_score']:.3f}\n")
                f.write(f"   用户保真度: {stats['user_fidelity']:.3f}\n")
                f.write(f"   模型效率: {stats['model_efficiency']:.3f}\n\n")
        
        print(f"📄 验证报告保存到: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="VQ-VAE + Transformer 验证器")
    
    # 路径参数
    parser.add_argument("--model_dir", type=str, required=True,
                       help="模型目录路径")
    parser.add_argument("--real_data_dir", type=str, required=True,
                       help="真实数据目录")
    parser.add_argument("--generated_data_dir", type=str, required=True,
                       help="生成数据目录")
    
    # 验证参数
    parser.add_argument("--target_user_id", type=int, default=0,
                       help="目标用户ID")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="相似性阈值")
    parser.add_argument("--device", type=str, default="auto",
                       help="计算设备")
    
    args = parser.parse_args()
    
    print("🔍 VQ-VAE + Transformer 验证器")
    print("=" * 50)
    
    # 创建验证器
    validator = VQVAEValidator(args.model_dir, args.device)
    
    # 运行综合验证
    results = validator.validate_comprehensive(
        real_data_dir=args.real_data_dir,
        generated_data_dir=args.generated_data_dir,
        target_user_id=args.target_user_id,
        threshold=args.threshold,
    )
    
    print(f"\n✅ 验证完成!")
    
    # 显示关键结果
    if 'quality_stats' in results:
        quality = results['quality_stats']
        print(f"\n📊 关键指标:")
        print(f"   综合质量分数: {quality['overall_score']:.3f}")
        
        if quality['overall_score'] > 0.7:
            print("🎉 优秀! VQ-VAE + Transformer表现出色")
        elif quality['overall_score'] > 0.5:
            print("👍 良好! 模型性能可接受")
        else:
            print("⚠️ 需要改进! 建议调整模型参数或训练策略")

if __name__ == "__main__":
    main()
