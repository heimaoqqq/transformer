#!/usr/bin/env python3
"""
上传项目到GitHub的脚本
使用SSH连接: git@github.com:heimaoqqq/VAE.git
"""

import os
import sys
import subprocess
from pathlib import Path

def run_git_command(cmd, description="", check_output=False):
    """运行Git命令"""
    print(f"🔄 {description}")
    print(f"   命令: {cmd}")
    
    try:
        if check_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        else:
            result = subprocess.run(cmd, shell=True, check=True)
            print(f"✅ {description} 成功")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"   错误: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ {description} 异常: {e}")
        return False

def check_git_status():
    """检查Git状态"""
    print("🔍 检查Git状态...")
    
    # 检查是否在Git仓库中
    if not Path('.git').exists():
        print("❌ 当前目录不是Git仓库")
        return False
    
    # 检查工作目录状态
    try:
        status = run_git_command("git status --porcelain", "检查工作目录状态", check_output=True)
        if status:
            print(f"📝 发现未提交的更改:")
            print(status)
        else:
            print("✅ 工作目录干净")
        return True
    except:
        return False

def setup_git_remote():
    """设置Git远程仓库"""
    print("🔗 设置Git远程仓库...")
    
    # 检查是否已有远程仓库
    try:
        remotes = run_git_command("git remote -v", "检查远程仓库", check_output=True)
        if "origin" in remotes:
            print("✅ 远程仓库已存在")
            print(remotes)
            return True
    except:
        pass
    
    # 添加远程仓库
    remote_url = "git@github.com:heimaoqqq/transformer.git"
    return run_git_command(f"git remote add origin {remote_url}", f"添加远程仓库 {remote_url}")

def add_and_commit_files():
    """添加并提交文件"""
    print("📁 添加并提交文件...")
    
    # 添加所有文件
    if not run_git_command("git add .", "添加所有文件"):
        return False
    
    # 检查是否有文件需要提交
    try:
        status = run_git_command("git status --porcelain --cached", "检查暂存区", check_output=True)
        if not status:
            print("ℹ️ 没有文件需要提交")
            return True
    except:
        pass
    
    # 提交文件
    commit_message = """🎨 完整的VQ-VAE+Transformer项目

✨ 新功能:
- 统一环境配置: setup_unified_environment.py
- 完整API兼容性检查: test_api_compatibility.py
- 统一环境测试: test_unified_environment.py

🔧 优化:
- 简化VQ-VAE模型导入逻辑
- 更新README.md，统一环境优先
- 智能版本选择，确保VQModel可用

📦 环境支持:
- 主推统一环境 (diffusers官方配置)
- 保留分阶段训练作为备选
- 完整的API兼容性验证

🎯 项目特点:
- 基于diffusers VQModel和transformers GPT2
- 支持微多普勒时频图生成
- 用户特征条件控制
- 8GB GPU即可训练"""
    
    return run_git_command(f'git commit -m "{commit_message}"', "提交更改")

def push_to_github():
    """推送到GitHub"""
    print("🚀 推送到GitHub...")
    
    # 检查当前分支
    try:
        branch = run_git_command("git branch --show-current", "检查当前分支", check_output=True)
        print(f"📍 当前分支: {branch}")
    except:
        branch = "main"
    
    # 推送到远程仓库
    return run_git_command(f"git push -u origin {branch}", f"推送到远程仓库 ({branch})")

def create_project_summary():
    """创建项目总结文件"""
    print("📄 创建项目总结...")
    
    summary = """# VQ-VAE + Transformer 微多普勒时频图生成项目

## 🎯 项目概述
基于diffusers VQModel和transformers GPT2的微多普勒时频图生成项目，支持用户特征条件控制。

## 🚀 快速开始

### 统一环境训练 (推荐)
```bash
# 1. 配置环境
python setup_unified_environment.py

# 2. API兼容性检查
python test_api_compatibility.py

# 3. 环境测试
python test_unified_environment.py

# 4. 开始训练
python train_main.py --data_dir /path/to/dataset
```

### 分阶段训练 (备选)
```bash
# VQ-VAE阶段
python setup_vqvae_environment.py
python train_main.py --skip_transformer --data_dir /path/to/dataset

# Transformer阶段 (重启后)
python setup_transformer_environment.py
python train_main.py --skip_vqvae --data_dir /path/to/dataset
```

## 📦 核心文件

### 环境配置
- `setup_unified_environment.py` - 统一环境配置 (推荐)
- `setup_vqvae_environment.py` - VQ-VAE专用环境
- `setup_transformer_environment.py` - Transformer专用环境

### 测试验证
- `test_api_compatibility.py` - 完整API兼容性检查
- `test_unified_environment.py` - 统一环境测试
- `test_cross_environment_compatibility.py` - 跨环境兼容性测试

### 训练脚本
- `train_main.py` - 主训练脚本
- `training/train_vqvae.py` - VQ-VAE专用训练
- `training/train_transformer.py` - Transformer专用训练

### 模型定义
- `models/vqvae_model.py` - 自定义VQ-VAE模型
- `models/transformer_model.py` - 自定义Transformer模型

## 🔧 技术特点

### 环境管理
- ✅ 统一环境: 使用diffusers官方配置
- ✅ 智能版本选择: 自动选择最佳diffusers版本
- ✅ API兼容性验证: 完整的兼容性检查
- ✅ 分阶段备选: 特殊情况下的解决方案

### 模型架构
- 🎨 VQ-VAE: 基于diffusers VQModel，支持图像离散化
- 🤖 Transformer: 基于transformers GPT2，支持序列生成
- 🎯 条件控制: 用户特征条件生成
- 💾 跨环境兼容: VQ-VAE模型支持跨环境使用

### 训练优化
- 🚀 低GPU要求: 8GB即可训练
- 📊 小数据友好: 离散化天然正则化
- ⚡ 灵活训练: 支持完整训练和分阶段训练
- 🔄 断点续训: 支持训练中断恢复

## 📋 系统要求
- Python 3.8+
- PyTorch 2.1.0+
- CUDA 12.1+ (推荐)
- GPU内存: 8GB+ (16GB推荐)

## 🎉 项目优势
1. **官方标准**: 遵循diffusers和transformers官方配置
2. **简化部署**: 统一环境减少配置复杂度
3. **完整验证**: 全面的API兼容性检查
4. **灵活训练**: 支持多种训练模式
5. **生产就绪**: 完整的错误处理和日志记录

## 📞 联系方式
- GitHub: https://github.com/heimaoqqq/VAE
- 项目地址: git@github.com:heimaoqqq/VAE.git
"""
    
    try:
        with open("PROJECT_SUMMARY.md", "w", encoding="utf-8") as f:
            f.write(summary)
        print("✅ 项目总结创建成功: PROJECT_SUMMARY.md")
        return True
    except Exception as e:
        print(f"❌ 项目总结创建失败: {e}")
        return False

def main():
    """主函数"""
    print("🎨 上传VQ-VAE+Transformer项目到GitHub")
    print("=" * 60)
    print("🎯 目标仓库: git@github.com:heimaoqqq/transformer.git")
    
    # 检查当前目录
    current_dir = Path.cwd()
    print(f"📍 当前目录: {current_dir}")
    
    # 切换到项目根目录
    project_root = Path(__file__).parent
    os.chdir(project_root)
    print(f"📁 切换到项目目录: {project_root}")
    
    steps = [
        ("检查Git状态", check_git_status),
        ("创建项目总结", create_project_summary),
        ("设置远程仓库", setup_git_remote),
        ("添加并提交文件", add_and_commit_files),
        ("推送到GitHub", push_to_github),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"\n❌ {step_name}失败，停止上传")
            return False
    
    print("\n🎉 项目上传成功！")
    print("✅ 所有文件已推送到GitHub")
    print("\n📋 下一步:")
    print("   1. 访问: https://github.com/heimaoqqq/transformer")
    print("   2. 检查项目文件是否完整")
    print("   3. 查看README.md了解使用方法")
    print("\n🚀 开始使用:")
    print("   git clone git@github.com:heimaoqqq/transformer.git")
    print("   cd transformer")
    print("   python setup_unified_environment.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
