#!/usr/bin/env python3
"""
条件扩散验证使用示例
演示如何验证31位用户步态微多普勒时频图像的条件扩散效果
"""

import os
import subprocess
from pathlib import Path

def run_command(cmd, description=""):
    """运行命令并显示结果"""
    print(f"\n🔄 {description}")
    print(f"命令: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - 完成")
            return True
        else:
            print(f"❌ {description} - 失败")
            return False
    except Exception as e:
        print(f"❌ {description} - 异常: {e}")
        return False

def main():
    """主函数 - 完整的验证流程示例"""
    
    print("🎯 条件扩散验证完整流程示例")
    print("=" * 60)
    
    # 配置路径 (请根据实际情况修改)
    data_dir = "/kaggle/input/dataset"  # 真实数据目录
    output_dir = "./validation_results"  # 验证结果输出目录
    model_dir = "/kaggle/working/outputs/vae_diffusion"  # 训练好的模型目录
    
    print(f"📊 数据目录: {data_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🤖 模型目录: {model_dir}")
    
    # 步骤1: 训练所有用户的分类器
    print(f"\n" + "="*60)
    print(f"步骤1: 训练31个用户分类器")
    print(f"="*60)
    
    train_cmd = f"""python validation_simple.py \
        --data_dir "{data_dir}" \
        --action train \
        --output_dir "{output_dir}" \
        --epochs 30 \
        --batch_size 32 \
        --max_samples 500"""
    
    if not run_command(train_cmd, "训练用户分类器"):
        print("❌ 分类器训练失败，停止流程")
        return
    
    # 步骤2: 生成测试图像 (使用现有的生成脚本)
    print(f"\n" + "="*60)
    print(f"步骤2: 生成测试图像")
    print(f"="*60)
    
    # 为几个用户生成图像进行测试
    test_users = [1, 5, 10, 15, 20, 25, 31]  # 选择几个代表性用户
    
    for user_id in test_users:
        print(f"\n🎨 为用户 {user_id} 生成图像...")
        
        generated_dir = f"{output_dir}/generated_user_{user_id:02d}"
        
        # 使用支持指导强度的生成脚本
        guidance_scale = 1.5  # 轻微CFG增强，提升条件控制

        generate_cmd = f"""python generate_with_guidance.py \
            --vae_path "{model_dir}/vae/final_model" \
            --unet_path "{model_dir}/unet/final_model" \
            --condition_encoder_path "{model_dir}/condition_encoder/final_model.pth" \
            --user_ids {user_id} \
            --num_images_per_user 50 \
            --num_inference_steps 50 \
            --guidance_scale {guidance_scale} \
            --output_dir "{generated_dir}" \
            --data_dir "{data_dir}" """
        
        if run_command(generate_cmd, f"生成用户 {user_id} 的图像"):
            print(f"  ✅ 用户 {user_id} 图像生成完成: {generated_dir}")
        else:
            print(f"  ❌ 用户 {user_id} 图像生成失败")
            continue
        
        # 步骤3: 验证生成图像
        print(f"\n🔍 验证用户 {user_id} 的生成图像...")
        
        # 3a. 单用户验证
        validate_cmd = f"""python validation_simple.py \
            --data_dir "{data_dir}" \
            --action validate \
            --output_dir "{output_dir}" \
            --generated_images_dir "{generated_dir}" \
            --target_user_id {user_id} \
            --confidence_threshold 0.8"""
        
        run_command(validate_cmd, f"单用户验证 - 用户 {user_id}")
        
        # 3b. 交叉验证 (最重要的测试)
        cross_validate_cmd = f"""python validation_simple.py \
            --data_dir "{data_dir}" \
            --action cross_validate \
            --output_dir "{output_dir}" \
            --generated_images_dir "{generated_dir}" \
            --target_user_id {user_id} \
            --confidence_threshold 0.8"""
        
        run_command(cross_validate_cmd, f"交叉验证 - 用户 {user_id}")
    
    # 步骤4: 分析总体结果
    print(f"\n" + "="*60)
    print(f"步骤4: 分析验证结果")
    print(f"="*60)
    
    print(f"\n📊 验证完成! 请查看以下文件:")
    print(f"  📁 输出目录: {output_dir}")
    print(f"  📄 分类器训练结果: {output_dir}/classifier_training_results.json")
    
    for user_id in test_users:
        print(f"  📄 用户 {user_id:2d} 交叉验证: {output_dir}/cross_validation_user_{user_id:02d}.json")
    
    print(f"\n💡 结果解读:")
    print(f"  1. 查看 classifier_training_results.json 确认分类器训练质量")
    print(f"  2. 查看各用户的 cross_validation_user_XX.json:")
    print(f"     - condition_effective: true 表示条件控制有效")
    print(f"     - discrimination_score > 0.3 表示区分度良好")
    print(f"     - target_user_performance.success_rate > 0.7 表示目标用户识别良好")
    print(f"     - other_users_performance.avg_success_rate < 0.3 表示其他用户正确拒绝")
    print(f"  3. 指导强度效果:")
    print(f"     - guidance_scale=1.0: 纯条件生成 (与训练时相同)")
    print(f"     - guidance_scale=1.5-2.0: 轻微CFG增强 (推荐)")
    print(f"     - guidance_scale>3.0: 强CFG增强 (可能过饱和)")
    
    print(f"\n🎉 条件扩散验证流程完成!")

def quick_test():
    """快速测试单个用户"""
    print("🚀 快速测试模式")
    print("=" * 40)
    
    # 配置 (请修改为实际路径)
    data_dir = "/kaggle/input/dataset"
    output_dir = "./quick_test"
    test_user_id = 1
    
    print(f"测试用户: {test_user_id}")
    
    # 1. 训练该用户的分类器
    train_cmd = f"""python validation_simple.py \
        --data_dir "{data_dir}" \
        --action train \
        --output_dir "{output_dir}" \
        --epochs 20 \
        --batch_size 16 \
        --max_samples 200"""
    
    if run_command(train_cmd, "快速训练分类器"):
        print("✅ 分类器训练完成")
    else:
        print("❌ 分类器训练失败")
        return
    
    # 2. 假设已有生成图像，进行验证
    generated_dir = f"{output_dir}/generated_user_{test_user_id:02d}"
    
    print(f"\n💡 请将用户 {test_user_id} 的生成图像放在: {generated_dir}")
    print(f"然后运行以下命令进行验证:")
    
    validate_cmd = f"""python validation_simple.py \
        --data_dir "{data_dir}" \
        --action cross_validate \
        --output_dir "{output_dir}" \
        --generated_images_dir "{generated_dir}" \
        --target_user_id {test_user_id}"""
    
    print(f"\n验证命令:")
    print(f"{validate_cmd}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        main()
