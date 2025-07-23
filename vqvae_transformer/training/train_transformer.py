#!/usr/bin/env python3
"""
Transformer训练脚本
第二阶段：训练Transformer学习从用户ID生成token序列
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from models.transformer_model import MicroDopplerTransformer
from models.vqvae_model import MicroDopplerVQVAE
from utils.data_loader import MicroDopplerDataset

class TransformerTrainer:
    """Transformer训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🤖 Transformer训练器初始化")
        print(f"   设备: {self.device}")
        print(f"   VQ-VAE路径: {args.vqvae_path}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   批次大小: {args.batch_size}")
        
        # 加载VQ-VAE模型
        self.vqvae_model = self._load_vqvae_model()
        
        # 创建Transformer模型
        self.transformer_model = self._create_transformer_model()
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.transformer_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 创建带warmup的学习率调度器
        self.warmup_steps = 1000  # 约2个epoch
        self.current_step = 0
        # total_steps将在train()方法中计算

        # 使用CosineAnnealingLR作为主调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.num_epochs
        )
        
    def _load_vqvae_model(self):
        """加载预训练的VQ-VAE模型"""
        vqvae_path = Path(self.args.vqvae_path)

        # 检查是否直接包含diffusers格式文件 (config.json + safetensors)
        config_file = vqvae_path / "config.json"
        safetensors_file = vqvae_path / "diffusion_pytorch_model.safetensors"

        if config_file.exists() and safetensors_file.exists():
            print(f"📂 加载VQ-VAE模型 (直接diffusers格式): {vqvae_path}")
            try:
                from models.vqvae_model import MicroDopplerVQVAE
                vqvae_model = MicroDopplerVQVAE.from_pretrained(vqvae_path)
                vqvae_model.to(self.device)
                vqvae_model.eval()
                print("✅ 成功加载直接diffusers格式模型")
                return vqvae_model
            except Exception as e:
                print(f"⚠️ 直接diffusers格式加载失败: {e}")
                print("🔄 尝试final_model子目录...")

        # 尝试final_model子目录 (diffusers格式)
        final_model_path = vqvae_path / "final_model"
        if final_model_path.exists():
            print(f"📂 加载VQ-VAE模型 (final_model子目录): {final_model_path}")
            try:
                from models.vqvae_model import MicroDopplerVQVAE
                vqvae_model = MicroDopplerVQVAE.from_pretrained(final_model_path)
                vqvae_model.to(self.device)
                vqvae_model.eval()
                print("✅ 成功加载final_model子目录格式模型")
                return vqvae_model
            except Exception as e:
                print(f"⚠️ final_model子目录格式加载失败: {e}")
                print("🔄 尝试checkpoint格式...")

        # 备选：使用checkpoint文件
        best_model_path = vqvae_path / "best_model.pth"
        if best_model_path.exists():
            model_file = best_model_path
        else:
            # 查找其他checkpoint文件
            model_files = list(vqvae_path.glob("*.pth"))
            if not model_files:
                raise FileNotFoundError(f"在 {vqvae_path} 中未找到VQ-VAE模型文件")
            model_file = model_files[0]

        print(f"📂 加载VQ-VAE模型 (checkpoint格式): {model_file}")

        # 加载checkpoint
        checkpoint = torch.load(model_file, map_location=self.device)

        # 重建VQ-VAE模型
        from models.vqvae_model import MicroDopplerVQVAE
        vqvae_model = MicroDopplerVQVAE(
            num_vq_embeddings=checkpoint['args'].codebook_size,
            commitment_cost=checkpoint['args'].commitment_cost,
            ema_decay=getattr(checkpoint['args'], 'ema_decay', 0.99),
        )

        # 加载权重
        vqvae_model.load_state_dict(checkpoint['model_state_dict'])
        vqvae_model.to(self.device)
        vqvae_model.eval()
        print("✅ 成功加载checkpoint格式模型")
        
        print(f"✅ VQ-VAE模型加载成功")
        return vqvae_model
        
    def _create_transformer_model(self):
        """创建Transformer模型"""
        model = MicroDopplerTransformer(
            vocab_size=self.args.codebook_size,
            max_seq_len=self.args.resolution * self.args.resolution // 16,  # VQ-VAE实际是4倍下采样: (128//4)^2 = 32^2 = 1024
            num_users=self.args.num_users,
            n_embd=self.args.n_embd,
            n_layer=self.args.n_layer,
            n_head=self.args.n_head,
            dropout=0.1,
            use_cross_attention=True,
        )
        model.to(self.device)

        print(f"✅ Transformer模型创建成功")
        print(f"   词汇表大小: {self.args.codebook_size}")
        print(f"   嵌入维度: {self.args.n_embd}")
        print(f"   层数: {self.args.n_layer}")
        print(f"   注意力头数: {self.args.n_head}")
        print(f"   序列长度: {model.max_seq_len}")
        print(f"   用户数量: {self.args.num_users}")

        # 测试增强功能是否工作
        self._test_enhanced_features(model)

        return model

    def _test_enhanced_features(self, model):
        """测试增强功能是否正确工作"""
        print(f"🧪 测试增强功能:")

        # 创建测试数据
        test_user_ids = torch.tensor([1, 2], device=self.device)
        test_tokens = torch.randint(0, 1024, (2, 1024), device=self.device)

        # 测试用户编码器
        with torch.no_grad():
            user_embeds = model.user_encoder(test_user_ids)
            print(f"   用户嵌入形状: {user_embeds.shape} (应该是[2, 512])")

            # 测试prepare_inputs
            inputs_dict = model.prepare_inputs(test_user_ids, test_tokens)

            print(f"   输入序列形状: {inputs_dict['input_ids'].shape}")
            print(f"   标签形状: {inputs_dict['labels'].shape}")

            if inputs_dict['encoder_hidden_states'] is not None:
                print(f"   交叉注意力状态形状: {inputs_dict['encoder_hidden_states'].shape} (应该是[2, 8, 512])")
                print(f"   注意力掩码形状: {inputs_dict['encoder_attention_mask'].shape}")
            else:
                print(f"   交叉注意力: 未使用")

        print(f"✅ 增强功能测试完成")

        # 性能基准测试
        self._benchmark_performance(model)

        return model

    def _benchmark_performance(self, model):
        """性能基准测试 - 验证增强功能的实际影响"""
        print(f"⚡ 性能基准测试:")

        import time
        import torch.profiler

        # 创建测试数据
        test_user_ids = torch.tensor([1, 2, 3, 4], device=self.device)
        test_tokens = torch.randint(0, 1024, (4, 1024), device=self.device)

        # 预热GPU
        for _ in range(3):
            with torch.no_grad():
                _ = model(test_user_ids, test_tokens)

        torch.cuda.synchronize()

        # 测试前向传播时间
        num_runs = 10
        start_time = time.time()

        for _ in range(num_runs):
            with torch.no_grad():
                outputs = model(test_user_ids, test_tokens)

        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs

        # 测试显存使用
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(self.device)

        outputs = model(test_user_ids, test_tokens)
        peak_memory = torch.cuda.max_memory_allocated(self.device)

        memory_used = (peak_memory - initial_memory) / 1024**2  # MB

        print(f"   前向传播时间: {avg_time*1000:.2f}ms")
        print(f"   显存使用: {memory_used:.1f}MB")
        print(f"   输出损失: {outputs.loss.item():.4f}")

        # 验证交叉注意力是否真正使用
        self._verify_cross_attention_usage(model, test_user_ids, test_tokens)

    def _verify_cross_attention_usage(self, model, test_user_ids, test_tokens):
        """验证交叉注意力是否真正被使用"""
        print(f"🔍 验证交叉注意力使用:")

        # 准备输入
        inputs = model.prepare_inputs(test_user_ids, test_tokens)

        # 检查是否有encoder_hidden_states
        has_encoder_states = inputs['encoder_hidden_states'] is not None
        print(f"   encoder_hidden_states存在: {'✅' if has_encoder_states else '❌'}")

        if has_encoder_states:
            encoder_shape = inputs['encoder_hidden_states'].shape
            print(f"   encoder状态形状: {encoder_shape}")

            # 验证GPT2是否真正使用交叉注意力
            # 通过hook监控交叉注意力层的激活
            cross_attn_activations = []

            def hook_fn(module, input, output):
                cross_attn_activations.append(output[0].shape if isinstance(output, tuple) else output.shape)

            # 注册hook到GPT2的交叉注意力层
            hooks = []
            for name, module in model.transformer.transformer.h[0].named_modules():
                if 'crossattention' in name.lower():
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)

            # 执行前向传播
            with torch.no_grad():
                _ = model.transformer(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    encoder_hidden_states=inputs['encoder_hidden_states'],
                    encoder_attention_mask=inputs['encoder_attention_mask'],
                    labels=inputs['labels'],
                )

            # 清理hooks
            for hook in hooks:
                hook.remove()

            if cross_attn_activations:
                print(f"   交叉注意力激活: ✅ {len(cross_attn_activations)}层")
                print(f"   激活形状: {cross_attn_activations[0] if cross_attn_activations else 'N/A'}")
            else:
                print(f"   交叉注意力激活: ❌ 未检测到")

        # 对比有无用户条件的输出差异
        print(f"🔬 用户条件影响测试:")

        # 使用实际的用户ID范围 [1-31]
        # 相同用户的输出
        same_user_ids = torch.tensor([5, 5, 5, 5], device=self.device)
        with torch.no_grad():
            same_outputs = model(same_user_ids, test_tokens)

        # 不同用户的输出 - 使用分散的用户ID
        diff_user_ids = torch.tensor([1, 10, 20, 31], device=self.device)
        with torch.no_grad():
            diff_outputs = model(diff_user_ids, test_tokens)

        # 极端对比：用户1 vs 用户31
        extreme_user1 = torch.tensor([1, 1, 1, 1], device=self.device)
        extreme_user31 = torch.tensor([31, 31, 31, 31], device=self.device)
        with torch.no_grad():
            extreme_outputs1 = model(extreme_user1, test_tokens)
            extreme_outputs31 = model(extreme_user31, test_tokens)

        # 计算多种差异指标
        same_logits_std = same_outputs.logits.std().item()
        diff_logits_std = diff_outputs.logits.std().item()

        # 计算极端用户差异
        extreme_diff = torch.abs(extreme_outputs1.logits - extreme_outputs31.logits).mean().item()

        # 计算用户嵌入的差异
        user_embed1 = model.user_encoder(torch.tensor([1], device=self.device))
        user_embed31 = model.user_encoder(torch.tensor([31], device=self.device))
        user_embed_diff = torch.abs(user_embed1 - user_embed31).mean().item()

        print(f"   相同用户输出标准差: {same_logits_std:.4f}")
        print(f"   不同用户输出标准差: {diff_logits_std:.4f}")
        print(f"   极端用户输出差异(1 vs 31): {extreme_diff:.4f}")
        print(f"   用户嵌入差异(1 vs 31): {user_embed_diff:.4f}")
        print(f"   用户缩放因子: {model.user_scale_factor.item():.4f}")

        # 更严格的判断标准
        is_significant = (diff_logits_std > same_logits_std * 1.05) or (extreme_diff > 0.01)
        print(f"   用户条件影响: {'✅显著' if is_significant else '❌微弱'}")

        return model

    def _update_learning_rate(self):
        """更新学习率，包含warmup机制"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Warmup阶段：线性增长
            warmup_factor = self.current_step / self.warmup_steps
            current_lr = self.args.learning_rate * warmup_factor

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            if self.current_step % 200 == 0:  # 每200步打印一次
                print(f"   Warmup步骤 {self.current_step}/{self.warmup_steps}, LR: {current_lr:.6f}")
        else:
            # Warmup完成后，使用cosine annealing
            # 简化：每个epoch结束时调用scheduler.step()
            pass  # scheduler.step()将在epoch结束时调用

    def _check_user_distribution(self, train_dataset, val_dataset, full_dataset):
        """检查训练集和验证集的用户分布"""
        print(f"👥 检查用户分布:")

        # 获取用户分布 - 检查更多样本以确保准确性
        def get_user_ids(dataset, max_samples=500):
            user_ids = set()
            # 使用步长采样，确保覆盖整个数据集
            step = max(1, len(dataset) // max_samples)
            indices = list(range(0, len(dataset), step))

            for i in indices:
                try:
                    sample = dataset[i]

                    # 处理不同的数据格式
                    if isinstance(sample, dict):
                        user_id = sample['user_id']
                    elif isinstance(sample, (list, tuple)) and len(sample) == 2:
                        _, user_id = sample
                    else:
                        continue

                    user_ids.add(user_id.item() if hasattr(user_id, 'item') else user_id)
                except Exception as e:
                    continue
            return user_ids

        train_users = get_user_ids(train_dataset)
        val_users = get_user_ids(val_dataset)

        print(f"   训练集用户: {len(train_users)}个 {sorted(list(train_users))}")
        print(f"   验证集用户: {len(val_users)}个 {sorted(list(val_users))}")

        # 检查是否有用户缺失
        missing_in_train = val_users - train_users
        missing_in_val = train_users - val_users

        if missing_in_train:
            print(f"   ⚠️ 训练集缺少用户: {sorted(list(missing_in_train))}")
        if missing_in_val:
            print(f"   ⚠️ 验证集缺少用户: {sorted(list(missing_in_val))}")

        if not missing_in_train and not missing_in_val:
            print(f"   ✅ 训练集和验证集都包含所有用户")
        else:
            print(f"   ℹ️ 注意：如果上述警告出现，可能是采样检查的限制，实际分层划分已确保所有用户都被正确分配")

        print()

    def _stratified_split(self, dataset, train_ratio=0.8):
        """按用户分层划分数据集，确保每个用户的样本都按比例分配"""
        print(f"🔄 执行分层划分 (确保每个用户都在训练集和验证集中)...")

        # 收集每个用户的样本索引
        user_indices = {}
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]

                # 处理不同的数据格式
                if isinstance(sample, dict):
                    user_id = sample['user_id']
                elif isinstance(sample, (list, tuple)) and len(sample) == 2:
                    _, user_id = sample
                else:
                    print(f"⚠️ 未知的样本格式: {type(sample)}")
                    continue

                user_id = user_id.item() if hasattr(user_id, 'item') else user_id

                if user_id not in user_indices:
                    user_indices[user_id] = []
                user_indices[user_id].append(idx)
            except Exception as e:
                print(f"⚠️ 处理样本{idx}时出错: {e}")
                continue

        print(f"   发现 {len(user_indices)} 个用户")

        # 为每个用户分配样本到训练集和验证集
        train_indices = []
        val_indices = []

        import random
        random.seed(42)  # 固定随机种子

        for user_id, indices in user_indices.items():
            # 随机打乱该用户的样本
            indices = indices.copy()
            random.shuffle(indices)

            # 计算训练集样本数（至少1个）
            user_train_size = max(1, int(len(indices) * train_ratio))

            # 如果用户只有1个样本，放到训练集
            if len(indices) == 1:
                train_indices.extend(indices)
                print(f"   用户{user_id}: 1个样本 → 训练集")
            else:
                # 分配样本
                user_train_indices = indices[:user_train_size]
                user_val_indices = indices[user_train_size:]

                train_indices.extend(user_train_indices)
                val_indices.extend(user_val_indices)

                print(f"   用户{user_id}: {len(indices)}个样本 → 训练集{len(user_train_indices)}个, 验证集{len(user_val_indices)}个")

        # 随机打乱最终的索引列表
        random.shuffle(train_indices)
        random.shuffle(val_indices)

        print(f"✅ 分层划分完成")

        # 验证分层划分结果
        self._verify_stratified_split(train_indices, val_indices, user_indices)

        return train_indices, val_indices

    def _verify_stratified_split(self, train_indices, val_indices, user_indices):
        """验证分层划分的结果"""
        print(f"🔍 验证分层划分结果:")

        # 检查每个用户在训练集和验证集中的分布
        train_users = set()
        val_users = set()

        for user_id, indices in user_indices.items():
            user_train_count = len([idx for idx in indices if idx in train_indices])
            user_val_count = len([idx for idx in indices if idx in val_indices])

            if user_train_count > 0:
                train_users.add(user_id)
            if user_val_count > 0:
                val_users.add(user_id)

        print(f"   训练集包含用户: {len(train_users)}个 {sorted(list(train_users))}")
        print(f"   验证集包含用户: {len(val_users)}个 {sorted(list(val_users))}")

        # 检查缺失用户
        all_users = set(user_indices.keys())
        missing_in_train = all_users - train_users
        missing_in_val = all_users - val_users

        if missing_in_train:
            print(f"   ❌ 训练集缺少用户: {sorted(list(missing_in_train))}")
        if missing_in_val:
            print(f"   ⚠️ 验证集缺少用户: {sorted(list(missing_in_val))} (可能是只有1个样本的用户)")

        if not missing_in_train and not missing_in_val:
            print(f"   ✅ 完美：所有用户都在训练集和验证集中")
        elif not missing_in_train:
            print(f"   ✅ 良好：所有用户都在训练集中，{len(missing_in_val)}个用户只在训练集中")

        print()
        
    def train(self):
        """训练Transformer"""
        print(f"\n🚀 开始Transformer训练")
        print(f"   训练轮数: {self.args.num_epochs}")
        print(f"   学习率: {self.args.learning_rate}")
        print(f"   评估间隔: 每5个epoch")
        print(f"   可视化生成: 每5个epoch")
        
        # 创建图像变换 - 转换为张量
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((self.args.resolution, self.args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
        ])

        # 创建完整数据集
        full_dataset = MicroDopplerDataset(
            data_dir=self.args.data_dir,
            transform=transform,  # 需要变换将PIL图像转为张量
            return_user_id=True,  # 需要用户ID进行条件生成
        )

        # 分层划分训练集和验证集 (80% 训练, 20% 验证)
        # 确保每个用户的样本都按比例分配到训练集和验证集
        train_indices, val_indices = self._stratified_split(full_dataset, train_ratio=0.8)

        print(f"📊 数据集划分:")
        print(f"   总样本数: {len(full_dataset)}")
        print(f"   训练集: {len(train_indices)} ({len(train_indices)/len(full_dataset)*100:.1f}%)")
        print(f"   验证集: {len(val_indices)} ({len(val_indices)/len(full_dataset)*100:.1f}%)")

        # 创建子数据集
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

        # 检查用户分布
        self._check_user_distribution(train_dataset, val_dataset, full_dataset)

        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,  # 验证集不需要shuffle
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        # 检查VQ-VAE质量
        self._check_vqvae_quality(train_dataloader)

        print(f"🚀 开始训练Transformer模型...")

        best_loss = float('inf')
        best_psnr = 0.0

        for epoch in range(self.args.num_epochs):
            self.transformer_model.train()
            total_loss = 0
            
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # 处理不同的batch格式
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    user_ids = batch['user_id'].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, user_ids = batch
                    images = images.to(self.device)
                    user_ids = user_ids.to(self.device)
                else:
                    print(f"❌ 未知的batch格式: {type(batch)}")
                    continue

                # 用户ID范围[1,31]直接使用，嵌入层已调整为支持这个范围
                
                # 使用VQ-VAE编码图像为token序列
                with torch.no_grad():
                    encoded = self.vqvae_model.encode(images, return_dict=True)
                    tokens = encoded['encoding_indices']  # [B, H, W] - VQ-VAE输出的2D token map

                    # 检查token值范围
                    min_token = tokens.min().item()
                    max_token = tokens.max().item()
                    if min_token < 0 or max_token >= self.args.codebook_size:
                        print(f"❌ Token值超出范围: [{min_token}, {max_token}], 跳过此批次")
                        continue

                    # 展平为序列 [B, H*W] - 对于128x128图像，4倍下采样后是32x32=1024
                    batch_size = tokens.shape[0]
                    tokens = tokens.view(batch_size, -1)  # [B, 1024]
                
                # Transformer训练
                self.optimizer.zero_grad()
                
                # 准备输入和目标 - 确保长度匹配
                # MicroDopplerTransformer会在内部添加用户token并处理序列
                # 我们直接传递完整的token序列
                input_tokens = tokens  # 完整的token序列 [B, 1024]
                
                # 前向传播
                outputs = self.transformer_model(
                    user_ids=user_ids,
                    token_sequences=input_tokens
                )
                
                # 使用Transformer内部计算的损失
                loss = outputs.loss

                # 添加空间一致性损失
                spatial_loss = self._compute_spatial_consistency_loss(input_tokens)
                loss = loss + 0.1 * spatial_loss  # 权重0.1
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer_model.parameters(), 1.0)
                self.optimizer.step()

                # 更新学习率（包含warmup）
                self._update_learning_rate()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # 定期保存checkpoint（减少频率）
                if batch_idx % 1000 == 0 and batch_idx > 0:
                    self._save_checkpoint(epoch, batch_idx, loss.item())
            
            # Warmup完成后，每个epoch结束时更新cosine scheduler
            if self.current_step > self.warmup_steps:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"   Cosine LR: {current_lr:.6f}")

            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")

            # 每5个epoch进行评估和可视化
            if (epoch + 1) % 5 == 0:
                print(f"\n📊 第{epoch+1}轮评估:")

                # 评估模型（使用验证集）
                eval_metrics = self.evaluate(val_dataloader)
                print(f"   验证损失: {eval_metrics['loss']:.4f}")
                print(f"   PSNR: {eval_metrics['psnr']:.2f} dB")
                print(f"   评估样本数: {eval_metrics['num_samples']}")

                # 生成可视化样本
                self.generate_and_save_samples(epoch)

                # 保存最佳PSNR模型
                if eval_metrics['psnr'] > best_psnr:
                    best_psnr = eval_metrics['psnr']
                    self._save_best_model(epoch, eval_metrics['psnr'], eval_metrics['loss'])



                print()  # 空行分隔

            # 保存基于训练损失的最佳模型（备用）
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        # 保存最终模型
        self._save_model("final_model.pth", self.args.num_epochs-1, avg_loss)
        print(f"✅ Transformer训练完成")
        print(f"🏆 最佳PSNR: {best_psnr:.2f} dB")
        
    def _save_checkpoint(self, epoch, batch_idx, loss):
        """保存训练检查点"""
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': self.transformer_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'args': self.args,
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pth"
        torch.save(checkpoint, checkpoint_path)
        
    def _save_model(self, filename, epoch, loss):
        """保存模型"""
        model_data = {
            'epoch': epoch,
            'model_state_dict': self.transformer_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'args': self.args,
        }
        
        model_path = self.output_dir / filename
        torch.save(model_data, model_path)
        print(f"💾 模型已保存: {model_path}")

    def evaluate(self, dataloader):
        """评估模型性能"""
        self.transformer_model.eval()
        total_loss = 0.0
        num_batches = 0

        # 用于计算PSNR的样本
        original_images = []
        generated_images = []

        print("🔍 开始模型评估...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 50:  # 限制评估样本数量
                    break

                # 处理不同的batch格式
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    user_ids = batch['user_id'].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, user_ids = batch
                    images = images.to(self.device)
                    user_ids = user_ids.to(self.device)
                else:
                    print(f"❌ 未知的batch格式: {type(batch)}")
                    continue

                # 检查token值范围
                encoded = self.vqvae_model.encode(images, return_dict=True)
                tokens = encoded['encoding_indices']

                min_token = tokens.min().item()
                max_token = tokens.max().item()
                if min_token < 0 or max_token >= self.args.codebook_size:
                    continue

                # 展平tokens
                batch_size = tokens.shape[0]
                tokens = tokens.view(batch_size, -1)
                input_tokens = tokens

                # 计算损失
                outputs = self.transformer_model(
                    user_ids=user_ids,
                    token_sequences=input_tokens
                )

                total_loss += outputs.loss.item()
                num_batches += 1

                # 收集前几个样本用于PSNR计算
                if batch_idx < 5:
                    # 生成图像
                    generated_tokens = self._generate_images(user_ids[:4])
                    if generated_tokens is not None:
                        # 解码生成的tokens
                        generated_imgs = self._decode_tokens(generated_tokens)
                        if generated_imgs is not None:
                            original_images.append(images[:4].cpu())
                            generated_images.append(generated_imgs.cpu())

        self.transformer_model.train()

        avg_loss = total_loss / max(num_batches, 1)

        # 计算PSNR
        psnr = self._calculate_psnr(original_images, generated_images)

        return {
            'loss': avg_loss,
            'psnr': psnr,
            'num_samples': num_batches * self.args.batch_size
        }

    def _generate_images(self, user_ids, max_length=1024):
        """生成图像tokens"""
        try:
            with torch.no_grad():
                # 使用Transformer生成tokens
                batch_size = user_ids.shape[0]
                device = user_ids.device

                # 开始token（用户token）
                generated = torch.full((batch_size, 1), self.transformer_model.user_token_id, device=device)

                # 逐步生成
                for step in range(max_length):
                    # 准备输入
                    inputs = self.transformer_model.prepare_inputs(user_ids, None)

                    # 更新input_ids为当前生成的序列
                    inputs['input_ids'] = generated
                    inputs['attention_mask'] = torch.ones_like(generated)

                    # 前向传播
                    if self.transformer_model.use_cross_attention:
                        outputs = self.transformer_model.transformer(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            encoder_hidden_states=inputs['encoder_hidden_states'],
                            encoder_attention_mask=inputs['encoder_attention_mask'],
                        )
                    else:
                        outputs = self.transformer_model.transformer(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                        )

                    # 获取下一个token的logits - 使用更保守的策略
                    next_token_logits = outputs.logits[:, -1, :]

                    # 限制logits到有效的token范围 [0, codebook_size-1]
                    if next_token_logits.shape[-1] > self.args.codebook_size:
                        next_token_logits = next_token_logits[:, :self.args.codebook_size]

                    # 使用更低的温度以减少随机性
                    temperature = max(0.3, self.args.generation_temperature * 0.5)
                    next_token_logits = next_token_logits / temperature

                    # Top-k采样以避免极端token
                    k = min(50, next_token_logits.shape[-1] // 4)  # 限制选择范围
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, k, dim=-1)

                    # 在top-k中采样
                    top_k_probs = F.softmax(top_k_logits, dim=-1)
                    sampled_indices = torch.multinomial(top_k_probs, num_samples=1)  # [batch_size, 1]
                    next_token = torch.gather(top_k_indices, -1, sampled_indices)  # [batch_size, 1]

                    # 确保token在有效范围内
                    next_token = torch.clamp(next_token, 0, self.args.codebook_size - 1)

                    # 添加到生成序列
                    generated = torch.cat([generated, next_token], dim=1)

                    # 检查是否达到目标长度
                    if generated.shape[1] >= max_length + 1:
                        break

                # 移除用户token，返回图像tokens
                image_tokens = generated[:, 1:]  # 去掉第一个用户token

                # 确保形状正确
                if image_tokens.shape[1] < max_length:
                    # 填充到正确长度
                    padding = torch.zeros(batch_size, max_length - image_tokens.shape[1], device=device, dtype=torch.long)
                    image_tokens = torch.cat([image_tokens, padding], dim=1)
                elif image_tokens.shape[1] > max_length:
                    # 截断到正确长度
                    image_tokens = image_tokens[:, :max_length]

                return image_tokens

        except Exception as e:
            print(f"⚠️ 生成图像失败: {e}")
            return None

    def _decode_tokens(self, tokens):
        """将tokens解码为图像"""
        try:
            with torch.no_grad():
                # 确保tokens是正确的数据类型
                if tokens.dtype != torch.long:
                    tokens = tokens.long()

                # 检查和修复token值范围
                min_token = tokens.min().item()
                max_token = tokens.max().item()

                # 过滤掉特殊token（用户token ID = codebook_size）
                special_token_mask = tokens >= self.args.codebook_size
                if special_token_mask.any():
                    print(f"⚠️ 发现{special_token_mask.sum().item()}个特殊token，将替换为随机有效token")
                    # 将特殊token替换为随机的有效token
                    random_tokens = torch.randint(0, self.args.codebook_size,
                                                 special_token_mask.shape,
                                                 device=tokens.device)
                    tokens = torch.where(special_token_mask, random_tokens, tokens)

                # 再次检查范围
                min_token = tokens.min().item()
                max_token = tokens.max().item()
                if min_token < 0 or max_token >= self.args.codebook_size:
                    print(f"⚠️ Token值仍超出范围: [{min_token}, {max_token}]")
                    return None

                print(f"✅ Token范围正常: [{min_token}, {max_token}]")

                # 重塑为2D token map
                batch_size = tokens.shape[0]
                tokens_2d = tokens.view(batch_size, 32, 32)  # 32x32 token map

                # 将token indices转换为embeddings
                # 获取VQ-VAE的量化器
                quantizer = self.vqvae_model.quantize

                # 将token indices转换为embeddings
                # tokens_2d: [B, H, W] -> embeddings: [B, H, W, D]
                embeddings = quantizer.embedding(tokens_2d)  # [B, 32, 32, 256]

                # 转换为VQ-VAE期望的格式: [B, D, H, W]
                embeddings = embeddings.permute(0, 3, 1, 2)  # [B, 256, 32, 32]

                # 使用VQ-VAE解码 - 跳过重新量化！
                decoded_images = self.vqvae_model.decode(embeddings, force_not_quantize=True)

                return decoded_images

        except Exception as e:
            print(f"⚠️ 解码tokens失败: {e}")
            # 打印更多调试信息
            if 'tokens' in locals():
                print(f"   tokens形状: {tokens.shape}")
                print(f"   tokens类型: {tokens.dtype}")
                print(f"   tokens范围: [{tokens.min().item()}, {tokens.max().item()}]")
            return None

    def _calculate_psnr(self, original_images, generated_images):
        """计算PSNR"""
        if not original_images or not generated_images:
            return 0.0

        try:
            import torch.nn.functional as F

            # 合并所有图像
            orig = torch.cat(original_images, dim=0)
            gen = torch.cat(generated_images, dim=0)

            # 确保形状匹配
            if orig.shape != gen.shape:
                gen = F.interpolate(gen, size=orig.shape[-2:], mode='bilinear', align_corners=False)

            # 归一化到[0,1]
            orig = (orig + 1) / 2  # 从[-1,1]到[0,1]
            gen = (gen + 1) / 2

            # 计算MSE
            mse = F.mse_loss(gen, orig)

            # 计算PSNR
            if mse > 0:
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                return psnr.item()
            else:
                return 100.0  # 完美匹配

        except Exception as e:
            print(f"⚠️ PSNR计算失败: {e}")
            return 0.0

    def generate_and_save_samples(self, epoch, num_users=4):
        """生成并保存样本图像"""
        print(f"🎨 生成第{epoch+1}轮样本图像...")

        self.transformer_model.eval()

        try:
            # 选择不同的用户ID进行生成
            user_ids = torch.tensor([1, 8, 16, 31], device=self.device)[:num_users]

            with torch.no_grad():
                # 生成图像tokens
                generated_tokens = self._generate_images(user_ids)

                if generated_tokens is not None:
                    # 解码为图像
                    generated_images = self._decode_tokens(generated_tokens)

                    if generated_images is not None:
                        # 保存图像
                        self._save_sample_images(generated_images, user_ids, epoch)
                        print(f"✅ 样本图像已保存")
                    else:
                        print(f"❌ 图像解码失败")
                else:
                    print(f"❌ 图像生成失败")

        except Exception as e:
            print(f"❌ 生成样本失败: {e}")

        self.transformer_model.train()

    def _save_sample_images(self, images, user_ids, epoch):
        """保存样本图像"""
        import matplotlib.pyplot as plt
        import numpy as np

        # 创建样本目录
        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        # 转换图像格式
        images = images.cpu().numpy()
        images = (images + 1) / 2  # 从[-1,1]到[0,1]
        images = np.clip(images, 0, 1)

        # 创建网格图像
        fig, axes = plt.subplots(1, len(user_ids), figsize=(4*len(user_ids), 4))
        if len(user_ids) == 1:
            axes = [axes]

        for i, (img, user_id) in enumerate(zip(images, user_ids)):
            # 转换为灰度图像（如果是3通道）
            if img.shape[0] == 3:
                img = np.mean(img, axis=0)
            else:
                img = img[0]

            axes[i].imshow(img, cmap='viridis')
            axes[i].set_title(f'User {user_id.item()}')
            axes[i].axis('off')

        plt.tight_layout()

        # 保存图像
        save_path = samples_dir / f"epoch_{epoch+1:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📸 样本图像保存至: {save_path}")

    def _save_best_model(self, epoch, psnr, loss):
        """保存最佳模型，删除旧的最佳模型"""
        # 删除旧的最佳模型
        old_best_files = list(self.output_dir.glob("best_model_*.pth"))
        for old_file in old_best_files:
            try:
                old_file.unlink()
                print(f"🗑️ 删除旧的最佳模型: {old_file.name}")
            except Exception as e:
                print(f"⚠️ 删除旧模型失败: {e}")

        # 保存新的最佳模型
        model_data = {
            'epoch': epoch,
            'model_state_dict': self.transformer_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'psnr': psnr,
            'loss': loss,
            'args': self.args,
        }

        best_model_path = self.output_dir / f"best_model_epoch_{epoch+1:03d}_psnr_{psnr:.2f}.pth"
        torch.save(model_data, best_model_path)
        print(f"🏆 保存最佳模型: {best_model_path.name} (PSNR: {psnr:.2f} dB)")

    def _save_checkpoint_to_kaggle(self, epoch, loss):
        """保存checkpoint到Kaggle工作目录"""
        try:
            kaggle_output_dir = Path("/kaggle/working")
            if kaggle_output_dir.exists():
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': self.transformer_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': loss,
                    'args': self.args,
                }

                checkpoint_path = kaggle_output_dir / f"transformer_checkpoint_epoch_{epoch+1:03d}.pth"
                torch.save(checkpoint_data, checkpoint_path)
                print(f"💾 Checkpoint已保存: {checkpoint_path.name}")

                # 只保留最近的3个checkpoint
                checkpoints = sorted(kaggle_output_dir.glob("transformer_checkpoint_*.pth"))
                if len(checkpoints) > 3:
                    for old_checkpoint in checkpoints[:-3]:
                        old_checkpoint.unlink()
                        print(f"🗑️ 删除旧checkpoint: {old_checkpoint.name}")

        except Exception as e:
            print(f"⚠️ Kaggle checkpoint保存失败: {e}")

    def _compute_spatial_consistency_loss(self, tokens):
        """计算空间一致性损失，鼓励相邻token的相似性"""
        try:
            batch_size = tokens.shape[0]
            # 重塑为2D: [B, 32, 32]
            tokens_2d = tokens.view(batch_size, 32, 32).float()

            # 计算水平方向的差异
            horizontal_diff = torch.abs(tokens_2d[:, :, 1:] - tokens_2d[:, :, :-1])

            # 计算垂直方向的差异
            vertical_diff = torch.abs(tokens_2d[:, 1:, :] - tokens_2d[:, :-1, :])

            # 总的空间一致性损失（鼓励平滑性）
            spatial_loss = torch.mean(horizontal_diff) + torch.mean(vertical_diff)

            return spatial_loss

        except Exception as e:
            # 如果计算失败，返回0
            return torch.tensor(0.0, device=tokens.device)

    def _check_vqvae_quality(self, dataloader):
        """检查VQ-VAE的编码质量和码本使用情况"""
        print(f"🔍 检查VQ-VAE质量:")

        self.vqvae_model.eval()
        all_tokens = []
        reconstruction_errors = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 10:  # 只检查前10个batch
                    break

                # 处理不同的batch格式
                if isinstance(batch, dict):
                    # 字典格式
                    images = batch['image'].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # 元组格式 (image, user_id)
                    images, user_ids = batch
                    images = images.to(self.device)
                else:
                    print(f"   ⚠️ 未知的batch格式: {type(batch)}")
                    continue

                # VQ-VAE编码和解码
                encoded = self.vqvae_model.encode(images, return_dict=True)

                # 处理编码结果
                if isinstance(encoded, dict):
                    latents = encoded['latents']
                    indices = encoded['encoding_indices']
                elif hasattr(encoded, 'latents'):
                    latents = encoded.latents
                    indices = getattr(encoded, 'encoding_indices', None)
                else:
                    latents = encoded
                    indices = None

                # 解码（跳过重新量化，因为latents已经是量化后的）
                reconstructed = self.vqvae_model.decode(latents, force_not_quantize=True)

                # 计算重建误差
                mse = torch.nn.functional.mse_loss(reconstructed, images)
                reconstruction_errors.append(mse.item())

                # 收集tokens
                if indices is not None:
                    all_tokens.extend(indices.flatten().cpu().numpy())

        # 分析结果
        avg_reconstruction_error = np.mean(reconstruction_errors)
        print(f"   平均重建误差: {avg_reconstruction_error:.6f}")

        if all_tokens:
            unique_tokens = len(set(all_tokens))
            total_possible = self.args.codebook_size
            usage_ratio = unique_tokens / total_possible
            print(f"   码本使用率: {unique_tokens}/{total_possible} ({usage_ratio:.2%})")

            # 检查token分布
            token_counts = np.bincount(all_tokens, minlength=total_possible)
            most_used = np.argmax(token_counts)
            least_used = np.argmin(token_counts[token_counts > 0]) if np.any(token_counts > 0) else 0

            print(f"   最常用token: {most_used} (使用{token_counts[most_used]}次)")
            print(f"   最少用token: {least_used} (使用{token_counts[least_used]}次)")

            # 警告
            if usage_ratio < 0.1:
                print(f"   ⚠️ 警告：码本使用率过低，可能导致生成多样性不足")
            if avg_reconstruction_error > 0.1:
                print(f"   ⚠️ 警告：重建误差过高，VQ-VAE质量可能有问题")

        self.vqvae_model.train()

def main():
    parser = argparse.ArgumentParser(description="Transformer训练脚本")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--vqvae_path", type=str, required=True, help="VQ-VAE模型路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    # 模型参数
    parser.add_argument("--resolution", type=int, default=128, help="图像分辨率")
    parser.add_argument("--codebook_size", type=int, default=1024, help="码本大小")
    parser.add_argument("--num_users", type=int, default=31, help="用户数量")

    # Transformer架构参数
    parser.add_argument("--n_embd", type=int, default=512, help="Transformer嵌入维度")
    parser.add_argument("--n_layer", type=int, default=8, help="Transformer层数")
    parser.add_argument("--n_head", type=int, default=8, help="注意力头数")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")

    # 保存和采样参数
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument("--sample_interval", type=int, default=10, help="样本生成间隔")
    parser.add_argument("--generation_temperature", type=float, default=1.0, help="生成温度")
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = TransformerTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
