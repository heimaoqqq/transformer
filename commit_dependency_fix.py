#!/usr/bin/env python3
"""
提交VQ-VAE依赖冲突修复
使用git_helper避免进程冲突
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from git_helper import run_git_command, check_git_status, safe_git_push

def main():
    """主函数"""
    print("🔧 提交VQ-VAE依赖冲突修复")
    print("=" * 50)
    
    # 1. 检查Git状态
    if not check_git_status():
        print("❌ Git状态检查失败")
        return False
    
    # 2. 添加修改的文件
    files_to_add = [
        "vqvae_transformer/setup_vqvae_environment.py",
        "vqvae_transformer/requirements.txt", 
        "vqvae_transformer/models/vqvae_model.py",
        "vqvae_transformer/README.md",
        "vqvae_transformer/test_vqvae_environment_fix.py",
        "vqvae_transformer/DEPENDENCY_FIX_SUMMARY.md",
        "vqvae_transformer/commit_dependency_fix.py"
    ]
    
    print("\n📁 添加修改的文件...")
    for file_path in files_to_add:
        if not run_git_command(f"git add {file_path}", f"添加 {file_path}"):
            print(f"⚠️ 添加 {file_path} 失败，继续...")
    
    # 3. 提交更改
    commit_message = """🔧 修复VQ-VAE环境依赖冲突 - 使用diffusers 0.24.0官方版本

✅ 核心修复:
- 使用diffusers 0.24.0官方VQModel版本
- 配置兼容的huggingface_hub版本范围
- 移除VQ-VAE阶段的transformers依赖

📦 版本组合:
- diffusers==0.24.0 (官方VQModel版本)
- huggingface_hub>=0.19.4,<0.26.0 (diffusers官方兼容范围)
- tokenizers>=0.14.1,<0.15.0, safetensors>=0.3.1

🔧 修改文件:
- setup_vqvae_environment.py: 更新版本组合和错误处理
- requirements.txt: 更新为兼容版本
- models/vqvae_model.py: 添加多路径导入支持
- README.md: 更新版本信息和说明

🧪 新增文件:
- test_vqvae_environment_fix.py: 环境验证脚本
- DEPENDENCY_FIX_SUMMARY.md: 详细修复总结

💡 技术要点:
- 避免transformers依赖冲突的根本原因
- 保持跨环境兼容性
- 支持分阶段训练策略"""
    
    print("\n💾 提交更改...")
    if not run_git_command(f'git commit -m "{commit_message}"', "提交VQ-VAE依赖修复"):
        print("❌ 提交失败")
        return False
    
    # 4. 推送到远程仓库
    print("\n🚀 推送到GitHub...")
    if not safe_git_push():
        print("❌ 推送失败")
        return False
    
    print("\n🎉 VQ-VAE依赖冲突修复已成功提交到GitHub!")
    print("\n📋 下一步:")
    print("   1. 验证修复: python test_vqvae_environment_fix.py")
    print("   2. 开始训练: python setup_vqvae_environment.py")
    print("   3. 查看总结: DEPENDENCY_FIX_SUMMARY.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
