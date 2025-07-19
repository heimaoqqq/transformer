#!/usr/bin/env python3
"""
现代化的条件扩散模型验证系统
参考成熟项目的设计模式，提供完整的验证流程
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from validation.user_classifier import UserValidationSystem

@dataclass
class ValidationConfig:
    """验证配置类 - 参考HuggingFace的配置模式"""
    # 基本配置
    target_user_id: int
    real_data_root: str
    output_dir: str = "./validation_results"
    
    # 分类器配置
    classifier_epochs: int = 30
    classifier_batch_size: int = 32
    classifier_lr: float = 5e-4
    max_samples_per_class: int = 1000
    confidence_threshold: float = 0.8
    
    # 生成配置
    num_images_to_generate: int = 100  # 增加到100张，获得更可靠的统计结果
    num_inference_steps: int = 50  # DDIM推理步数，建议50-200
    batch_size: int = 10  # 批量生成大小，充分利用显存
    
    # 模型路径
    vae_path: Optional[str] = None
    unet_path: Optional[str] = None
    condition_encoder_path: Optional[str] = None
    
    # 设备配置
    device: str = "auto"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

class ConditionalDiffusionValidator:
    """现代化的条件扩散模型验证器 - 参考Diffusers的Pipeline设计"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validation_system = UserValidationSystem(device=config.device)
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 模型组件 (延迟加载)
        self.vae = None
        self.unet = None
        self.condition_encoder = None
        self.scheduler = None
        self.user_id_mapping = None
        
    def load_models(self) -> bool:
        """加载所有必要的模型组件"""
        if not all([self.config.vae_path, self.config.unet_path, self.config.condition_encoder_path]):
            print("❌ 缺少模型路径，无法加载模型")
            return False
            
        try:
            print("📂 加载模型组件...")
            
            # 加载VAE
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(self.config.vae_path)
            self.vae = self.vae.to(self.config.device)
            print("  ✅ VAE加载完成")
            
            # 加载UNet
            from diffusers import UNet2DConditionModel
            self.unet = UNet2DConditionModel.from_pretrained(self.config.unet_path)
            self.unet = self.unet.to(self.config.device)
            print("  ✅ UNet加载完成")
            
            # 获取用户ID映射
            self.user_id_mapping = self._get_user_id_mapping()
            num_users = len(self.user_id_mapping)
            print(f"  📊 用户映射: {self.user_id_mapping}")
            
            # 加载条件编码器
            from training.train_diffusion import UserConditionEncoder
            self.condition_encoder = UserConditionEncoder(
                num_users=num_users,
                embed_dim=self.unet.config.cross_attention_dim
            )
            
            condition_encoder_state = torch.load(self.config.condition_encoder_path, map_location='cpu')
            self.condition_encoder.load_state_dict(condition_encoder_state)
            self.condition_encoder = self.condition_encoder.to(self.config.device)
            print("  ✅ 条件编码器加载完成")
            
            # 创建调度器 (与训练时一致)
            from diffusers import DDPMScheduler, DDIMScheduler
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                prediction_type="epsilon",
            )
            # 使用DDIM调度器进行推理 (与训练时生成样本一致)
            self.scheduler = DDIMScheduler.from_config(noise_scheduler.config)
            print("  ✅ 调度器创建完成")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_user_id_mapping(self) -> Dict[int, int]:
        """获取用户ID映射 - 与训练时保持一致，并进行一致性检查"""
        data_path = Path(self.config.real_data_root)
        all_users = []

        print(f"  🔍 扫描数据目录: {data_path}")

        for user_dir in data_path.iterdir():
            if user_dir.is_dir() and user_dir.name.startswith('ID_'):
                try:
                    user_id = int(user_dir.name.split('_')[1])
                    all_users.append(user_id)

                    # 检查图像数量
                    image_files = list(user_dir.glob("*.png")) + list(user_dir.glob("*.jpg"))
                    print(f"    ID_{user_id:2d}: {len(image_files):3d} 张图像")

                except ValueError:
                    print(f"    ⚠️  无效目录名: {user_dir.name}")
                    continue

        all_users = sorted(all_users)
        user_mapping = {user_id: idx for idx, user_id in enumerate(all_users)}

        print(f"  📊 用户映射 (训练时一致): {user_mapping}")

        # 检查目标用户是否存在
        if self.config.target_user_id not in user_mapping:
            print(f"  ❌ 目标用户 {self.config.target_user_id} 不在数据中!")
            print(f"  💡 可用用户: {sorted(user_mapping.keys())}")

        return user_mapping
    
    def train_classifier(self) -> bool:
        """训练用户分类器"""
        print(f"\n🤖 训练用户 {self.config.target_user_id} 的分类器")
        print(f"  参数: epochs={self.config.classifier_epochs}, batch_size={self.config.classifier_batch_size}")
        
        try:
            # 准备数据
            image_paths, labels = self._prepare_classifier_data()
            
            if len(image_paths) == 0:
                print(f"❌ 没有可用的训练数据")
                return False
            
            # 训练分类器
            history = self.validation_system.train_user_classifier(
                user_id=self.config.target_user_id,
                image_paths=image_paths,
                labels=labels,
                epochs=self.config.classifier_epochs,
                batch_size=self.config.classifier_batch_size,
                learning_rate=self.config.classifier_lr
            )
            
            # 保存训练曲线
            plot_path = self.output_path / f"user_{self.config.target_user_id:02d}_training.png"
            self.validation_system.plot_training_history(history, str(plot_path))
            
            # 检查训练效果
            best_val_acc = max(history['val_acc'])
            print(f"  📊 最佳验证准确率: {best_val_acc:.3f}")
            
            return best_val_acc > 0.7  # 设定最低准确率要求
            
        except Exception as e:
            print(f"❌ 分类器训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _prepare_classifier_data(self) -> Tuple[List[str], List[int]]:
        """准备分类器训练数据"""
        data_path = Path(self.config.real_data_root)
        target_user_dir = None
        other_user_dirs = []
        
        # 查找目标用户和其他用户目录
        for item in data_path.iterdir():
            if item.is_dir():
                if item.name == f"ID_{self.config.target_user_id}":
                    target_user_dir = item
                elif item.name.startswith("ID_"):
                    try:
                        other_user_id = int(item.name.split("_")[1])
                        if other_user_id != self.config.target_user_id:
                            other_user_dirs.append(item)
                    except ValueError:
                        continue
        
        if target_user_dir is None:
            print(f"❌ 未找到用户 {self.config.target_user_id} 的数据目录")
            return [], []

        # 使用改进的数据准备方法
        return self.validation_system.prepare_user_data(
            user_id=self.config.target_user_id,
            real_images_dir=str(target_user_dir),
            other_users_dirs=[str(d) for d in other_user_dirs],
            max_samples_per_class=self.config.max_samples_per_class,
            negative_ratio=3.0  # 负样本是正样本的3倍
        )

    def generate_images(self) -> Optional[str]:
        """生成指定用户的图像"""
        if not all([self.vae, self.unet, self.condition_encoder, self.scheduler]):
            print("❌ 模型未加载，无法生成图像")
            return None

        if self.config.target_user_id not in self.user_id_mapping:
            print(f"❌ 用户 {self.config.target_user_id} 不在映射中")
            return None

        print(f"\n🎨 生成用户 {self.config.target_user_id} 的图像")
        print(f"  参数: 纯条件生成, steps={self.config.num_inference_steps}")

        try:
            # 创建输出目录
            gen_output_dir = self.output_path / "generated_images" / f"user_{self.config.target_user_id:02d}"
            gen_output_dir.mkdir(parents=True, exist_ok=True)

            # 获取用户索引
            user_idx = self.user_id_mapping[self.config.target_user_id]

            # 设置调度器
            self.scheduler.set_timesteps(self.config.num_inference_steps)

            # 生成图像
            self.vae.eval()
            self.unet.eval()
            self.condition_encoder.eval()

            with torch.no_grad():
                # 批量生成配置
                batch_size = self.config.batch_size  # 使用配置中的批量大小
                total_images = self.config.num_images_to_generate
                num_batches = (total_images + batch_size - 1) // batch_size

                print(f"  📊 批量生成配置: {batch_size}张/批, 共{num_batches}批")

                image_count = 0
                for batch_idx in range(num_batches):
                    # 计算当前批次的实际大小
                    current_batch_size = min(batch_size, total_images - batch_idx * batch_size)
                    print(f"  🎨 生成批次 {batch_idx+1}/{num_batches} ({current_batch_size}张)...")

                    # 批量随机噪声
                    latents = torch.randn(current_batch_size, 4, 32, 32, device=self.config.device)

                    # 批量用户条件
                    user_tensor = torch.tensor([user_idx] * current_batch_size, device=self.config.device)
                    user_embedding = self.condition_encoder(user_tensor)

                    # 确保3D张量格式
                    if user_embedding.dim() == 2:
                        user_embedding = user_embedding.unsqueeze(1)

                    # 扩散过程
                    latents = latents * self.scheduler.init_noise_sigma

                    for t in self.scheduler.timesteps:
                        # 批量纯条件预测 (与训练时相同)
                        noise_pred = self.unet(
                            latents,
                            t,
                            encoder_hidden_states=user_embedding
                        ).sample

                        # 调度器步骤
                        latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                    # 批量解码为图像
                    vae_model = self.vae.module if hasattr(self.vae, 'module') else self.vae
                    latents = latents / vae_model.config.scaling_factor
                    images = vae_model.decode(latents).sample
                    images = images.clamp(0, 1)

                    # 批量保存图像
                    from PIL import Image
                    batch_images = images.cpu().permute(0, 2, 3, 1).numpy()

                    for i in range(current_batch_size):
                        image = (batch_images[i] * 255).astype(np.uint8)
                        pil_image = Image.fromarray(image)

                        save_path = gen_output_dir / f"user_{self.config.target_user_id}_generated_{image_count+1:02d}.png"
                        pil_image.save(save_path)
                        image_count += 1

            print(f"  ✅ 生成完成，保存在: {gen_output_dir}")
            return str(gen_output_dir)

        except Exception as e:
            print(f"❌ 图像生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def validate_generated_images(self, generated_images_dir: str) -> Dict:
        """验证生成图像 - 改进版本，包含对比控制实验"""
        print(f"\n🔍 验证生成图像 (改进版本)")

        try:
            # 1. 原有的分类器验证
            basic_result = self.validation_system.validate_generated_images(
                user_id=self.config.target_user_id,
                generated_images_dir=generated_images_dir,
                confidence_threshold=self.config.confidence_threshold
            )

            # 2. 对比控制实验
            control_result = self._controlled_validation_experiment(generated_images_dir)

            # 3. 全用户对比矩阵验证（可选，更全面）
            matrix_result = self._full_user_matrix_validation()

            # 4. 合并结果
            result = {
                'basic_validation': basic_result,
                'control_experiment': control_result,
                'user_matrix_validation': matrix_result,
                'overall_success': self._evaluate_overall_success(basic_result, control_result, matrix_result)
            }

            # 保存验证结果
            result_path = self.output_path / f"user_{self.config.target_user_id:02d}_validation.json"
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)

            return result

        except Exception as e:
            print(f"❌ 验证失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _controlled_validation_experiment(self, generated_images_dir: str) -> Dict:
        """对比控制实验 - 验证条件生成的有效性"""
        print(f"  🧪 执行对比控制实验...")

        try:
            # 1. 对比所有其他用户（更严格的验证）
            all_other_users = [uid for uid in self.user_id_mapping.keys()
                             if uid != self.config.target_user_id]

            # 如果用户太多，随机选择最多10个进行对比（平衡严格性和效率）
            import random
            if len(all_other_users) > 10:
                wrong_user_ids = random.sample(all_other_users, 10)
                print(f"    从{len(all_other_users)}个其他用户中随机选择10个进行对比")
            else:
                wrong_user_ids = all_other_users
                print(f"    对比所有{len(wrong_user_ids)}个其他用户")

            control_results = {}

            # 2. 为每个错误用户ID生成图像并验证
            for wrong_id in wrong_user_ids:
                print(f"    生成错误条件图像 (用户{wrong_id})...")

                # 生成错误条件的图像
                wrong_images_dir = self._generate_wrong_condition_images(wrong_id)

                if wrong_images_dir:
                    # 用目标用户的分类器验证错误条件图像
                    wrong_result = self.validation_system.validate_generated_images(
                        user_id=self.config.target_user_id,
                        generated_images_dir=wrong_images_dir,
                        confidence_threshold=self.config.confidence_threshold
                    )
                    control_results[f'wrong_user_{wrong_id}'] = wrong_result

            # 3. 计算对比指标
            if control_results:
                # 正确条件的结果
                correct_result = self.validation_system.validate_generated_images(
                    user_id=self.config.target_user_id,
                    generated_images_dir=generated_images_dir,
                    confidence_threshold=self.config.confidence_threshold
                )

                correct_success_rate = correct_result.get('success_rate', 0)
                wrong_success_rates = [r.get('success_rate', 0) for r in control_results.values()]
                avg_wrong_success_rate = sum(wrong_success_rates) / len(wrong_success_rates)

                # 条件控制效果：正确条件应该明显好于错误条件
                condition_control_ratio = correct_success_rate / (avg_wrong_success_rate + 1e-6)

                return {
                    'correct_condition_success_rate': correct_success_rate,
                    'wrong_conditions_avg_success_rate': avg_wrong_success_rate,
                    'condition_control_ratio': condition_control_ratio,
                    'control_effective': condition_control_ratio > 2.0,  # 正确条件应该至少好2倍
                    'detailed_wrong_results': control_results
                }
            else:
                return {'error': 'Failed to generate control images'}

        except Exception as e:
            print(f"    ❌ 对比实验失败: {e}")
            return {'error': str(e)}

    def _generate_wrong_condition_images(self, wrong_user_id: int, num_images: int = 4) -> str:
        """生成错误条件的图像用于对比"""
        try:
            # 创建错误条件图像的输出目录
            wrong_dir = self.output_path / "control_images" / f"wrong_user_{wrong_user_id}"
            wrong_dir.mkdir(parents=True, exist_ok=True)

            # 获取错误用户的索引
            wrong_user_idx = self.user_id_mapping[wrong_user_id]

            # 设置调度器
            self.scheduler.set_timesteps(self.config.num_inference_steps)

            # 生成图像
            self.vae.eval()
            self.unet.eval()
            self.condition_encoder.eval()

            with torch.no_grad():
                # 对比实验也使用批量生成（数量较少，一次性生成）
                print(f"    批量生成{num_images}张对比图像...")

                # 批量随机噪声
                latents = torch.randn(num_images, 4, 32, 32, device=self.config.device)

                # 批量错误用户条件
                user_tensor = torch.tensor([wrong_user_idx] * num_images, device=self.config.device)
                user_embedding = self.condition_encoder(user_tensor)

                # 确保3D张量格式
                if user_embedding.dim() == 2:
                    user_embedding = user_embedding.unsqueeze(1)

                # 扩散过程
                latents = latents * self.scheduler.init_noise_sigma

                for t in self.scheduler.timesteps:
                    # 批量纯条件预测
                    noise_pred = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=user_embedding
                    ).sample

                    # 调度器步骤
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # 批量解码为图像
                vae_model = self.vae.module if hasattr(self.vae, 'module') else self.vae
                latents = latents / vae_model.config.scaling_factor
                images = vae_model.decode(latents).sample
                images = images.clamp(0, 1)

                # 批量保存图像
                from PIL import Image
                batch_images = images.cpu().permute(0, 2, 3, 1).numpy()

                for i in range(num_images):
                    image = (batch_images[i] * 255).astype(np.uint8)
                    pil_image = Image.fromarray(image)

                    save_path = wrong_dir / f"wrong_condition_{i+1:02d}.png"
                    pil_image.save(save_path)

            return str(wrong_dir)

        except Exception as e:
            print(f"    ❌ 生成错误条件图像失败: {e}")
            return None

    def _full_user_matrix_validation(self) -> Dict:
        """全用户对比矩阵验证 - 最严格的验证方法"""
        print(f"  📊 执行全用户矩阵验证...")

        try:
            all_users = list(self.user_id_mapping.keys())
            target_user = self.config.target_user_id

            # 限制验证规模（避免计算量过大）
            if len(all_users) > 8:
                print(f"    用户数量({len(all_users)})较多，跳过矩阵验证以节省时间")
                return {'skipped': True, 'reason': 'too_many_users'}

            print(f"    为所有{len(all_users)}个用户生成图像并交叉验证...")

            # 生成所有用户的图像
            user_images = {}
            for user_id in all_users:
                print(f"      生成用户{user_id}的图像...")
                images_dir = self._generate_wrong_condition_images(user_id, num_images=2)
                if images_dir:
                    user_images[user_id] = images_dir

            # 用目标用户的分类器验证所有生成图像
            validation_matrix = {}
            for generated_user_id, images_dir in user_images.items():
                result = self.validation_system.validate_generated_images(
                    user_id=target_user,  # 始终用目标用户的分类器
                    generated_images_dir=images_dir,
                    confidence_threshold=self.config.confidence_threshold
                )
                validation_matrix[generated_user_id] = result.get('success_rate', 0)

            # 分析结果
            target_success_rate = validation_matrix.get(target_user, 0)
            other_success_rates = [rate for uid, rate in validation_matrix.items() if uid != target_user]

            if other_success_rates:
                avg_other_success_rate = sum(other_success_rates) / len(other_success_rates)
                max_other_success_rate = max(other_success_rates)

                # 计算各种对比指标
                avg_ratio = target_success_rate / (avg_other_success_rate + 1e-6)
                max_ratio = target_success_rate / (max_other_success_rate + 1e-6)

                # 严格标准：目标用户应该明显好于所有其他用户
                matrix_success = (
                    target_success_rate >= 0.6 and  # 目标用户成功率足够高
                    avg_ratio >= 2.0 and           # 平均比其他用户好2倍
                    max_ratio >= 1.5                # 比最好的其他用户也要好1.5倍
                )

                return {
                    'validation_matrix': validation_matrix,
                    'target_user_success_rate': target_success_rate,
                    'avg_other_success_rate': avg_other_success_rate,
                    'max_other_success_rate': max_other_success_rate,
                    'avg_ratio': avg_ratio,
                    'max_ratio': max_ratio,
                    'matrix_validation_success': matrix_success,
                    'criteria': {
                        'min_target_success_rate': 0.6,
                        'min_avg_ratio': 2.0,
                        'min_max_ratio': 1.5
                    }
                }
            else:
                return {'error': 'No other users to compare'}

        except Exception as e:
            print(f"    ❌ 矩阵验证失败: {e}")
            return {'error': str(e)}

    def _evaluate_overall_success(self, basic_result: Dict, control_result: Dict, matrix_result: Dict = None) -> Dict:
        """评估整体验证成功性"""
        # 基础验证指标
        basic_success_rate = basic_result.get('success_rate', 0)
        basic_avg_confidence = basic_result.get('avg_confidence', 0)

        # 对比控制指标
        control_effective = control_result.get('control_effective', False)
        condition_ratio = control_result.get('condition_control_ratio', 0)

        # 矩阵验证指标（如果可用）
        matrix_success = True  # 默认通过
        if matrix_result and 'matrix_validation_success' in matrix_result:
            matrix_success = matrix_result.get('matrix_validation_success', False)

        # 综合评估（更严格的标准）
        overall_success = (
            basic_success_rate >= 0.6 and
            basic_avg_confidence >= 0.7 and
            control_effective and
            matrix_success  # 如果有矩阵验证，也必须通过
        )

        result = {
            'overall_success': overall_success,
            'basic_success_rate': basic_success_rate,
            'basic_avg_confidence': basic_avg_confidence,
            'condition_control_effective': control_effective,
            'condition_control_ratio': condition_ratio,
            'evaluation_criteria': {
                'min_success_rate': 0.6,
                'min_avg_confidence': 0.7,
                'min_control_ratio': 2.0
            }
        }

        # 添加矩阵验证结果（如果有）
        if matrix_result and 'matrix_validation_success' in matrix_result:
            result['matrix_validation_success'] = matrix_success
            result['evaluation_criteria']['matrix_validation_required'] = True

        return result

    def run_full_pipeline(self, generate_images: bool = True) -> Dict:
        """运行完整的验证流程"""
        print(f"🚀 开始完整验证流程")
        print(f"目标用户: {self.config.target_user_id}")
        print(f"输出目录: {self.config.output_dir}")
        print("=" * 60)

        results = {
            "target_user_id": self.config.target_user_id,
            "config": self.config.__dict__,
            "classifier_trained": False,
            "images_generated": False,
            "validation_completed": False,
            "success": False
        }

        # 步骤1: 训练分类器
        if not self.train_classifier():
            print("❌ 分类器训练失败，终止流程")
            return results

        results["classifier_trained"] = True

        # 步骤2: 生成图像 (可选)
        generated_dir = None
        if generate_images:
            if not self.load_models():
                print("❌ 模型加载失败，跳过图像生成")
            else:
                generated_dir = self.generate_images()
                if generated_dir:
                    results["images_generated"] = True
                    results["generated_images_dir"] = generated_dir

        # 步骤3: 验证图像
        if generated_dir:
            validation_result = self.validate_generated_images(generated_dir)
            if validation_result:
                results["validation_completed"] = True
                results["validation_result"] = validation_result

                # 判断整体成功
                success_rate = validation_result.get('success_rate', 0)
                avg_confidence = validation_result.get('avg_confidence', 0)

                if success_rate >= 0.6 and avg_confidence >= 0.8:
                    results["success"] = True
                    print(f"🎉 验证成功! 成功率: {success_rate:.2f}, 平均置信度: {avg_confidence:.3f}")
                else:
                    print(f"⚠️  验证结果不理想. 成功率: {success_rate:.2f}, 平均置信度: {avg_confidence:.3f}")
                    print(f"💡 建议: 尝试更多推理步数 (num_inference_steps > {self.config.num_inference_steps})")

        return results

def main():
    """主函数 - 现代化的命令行接口"""
    parser = argparse.ArgumentParser(
        description="现代化的条件扩散模型验证系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必需参数
    parser.add_argument("--target_user_id", type=int, required=True,
                       help="目标用户ID")
    parser.add_argument("--real_data_root", type=str, required=True,
                       help="真实数据根目录")

    # 基本配置
    parser.add_argument("--output_dir", type=str, default="./validation_results",
                       help="输出目录")
    parser.add_argument("--device", type=str, default="auto",
                       help="计算设备 (auto/cuda/cpu)")

    # 分类器配置
    parser.add_argument("--classifier_epochs", type=int, default=30,
                       help="分类器训练轮数")
    parser.add_argument("--classifier_batch_size", type=int, default=32,
                       help="分类器批次大小")
    parser.add_argument("--classifier_lr", type=float, default=5e-4,
                       help="分类器学习率")
    parser.add_argument("--max_samples_per_class", type=int, default=1000,
                       help="每类最大样本数")
    parser.add_argument("--confidence_threshold", type=float, default=0.8,
                       help="置信度阈值")


    # 生成配置
    parser.add_argument("--generate_images", action="store_true",
                       help="是否生成图像")
    parser.add_argument("--num_images_to_generate", type=int, default=100,
                       help="生成图像数量 (建议100+张获得可靠统计结果)")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="DDIM推理步数 (建议50-200)")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="批量生成大小 (根据显存调整，建议8-16)")

    # 模型路径
    parser.add_argument("--vae_path", type=str,
                       help="VAE模型路径")
    parser.add_argument("--unet_path", type=str,
                       help="UNet模型路径")
    parser.add_argument("--condition_encoder_path", type=str,
                       help="条件编码器路径")

    args = parser.parse_args()

    # 创建配置
    config = ValidationConfig(
        target_user_id=args.target_user_id,
        real_data_root=args.real_data_root,
        output_dir=args.output_dir,
        classifier_epochs=args.classifier_epochs,
        classifier_batch_size=args.classifier_batch_size,
        classifier_lr=args.classifier_lr,
        max_samples_per_class=args.max_samples_per_class,
        confidence_threshold=args.confidence_threshold,
        num_images_to_generate=args.num_images_to_generate,
        num_inference_steps=args.num_inference_steps,
        batch_size=args.batch_size,
        vae_path=args.vae_path,
        unet_path=args.unet_path,
        condition_encoder_path=args.condition_encoder_path,
        device=args.device
    )

    # 打印配置
    print("🔧 验证配置:")
    print(f"  目标用户: {config.target_user_id}")
    print(f"  数据目录: {config.real_data_root}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  分类器: 标准ResNet-18, epochs={config.classifier_epochs}, batch_size={config.classifier_batch_size}")
    if args.generate_images:
        print(f"  生成: 纯条件生成, steps={config.num_inference_steps}, batch_size={config.batch_size}")
        print(f"  模型: VAE={config.vae_path is not None}, UNet={config.unet_path is not None}")
    print("=" * 60)

    # 创建验证器并运行
    validator = ConditionalDiffusionValidator(config)
    results = validator.run_full_pipeline(generate_images=args.generate_images)

    # 输出结果
    print(f"\n📋 验证结果总结:")
    print(f"  分类器训练: {'✅' if results['classifier_trained'] else '❌'}")
    if args.generate_images:
        print(f"  图像生成: {'✅' if results['images_generated'] else '❌'}")
        print(f"  验证完成: {'✅' if results['validation_completed'] else '❌'}")
        print(f"  整体成功: {'🎉' if results['success'] else '⚠️'}")

    # 保存完整结果
    result_file = Path(config.output_dir) / f"user_{config.target_user_id:02d}_complete_results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 完整结果保存在: {result_file}")

    if results.get('success'):
        print("🎉 验证成功完成!")
        return 0
    else:
        print("⚠️  验证未完全成功，请检查结果并调整参数")
        return 1

if __name__ == "__main__":
    exit(main())
