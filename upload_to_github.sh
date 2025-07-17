#!/bin/bash

# 微多普勒时频图数据增广项目 - GitHub上传脚本
# Repository: git@github.com:heimaoqqq/VAE.git

echo "🚀 开始上传微多普勒VAE项目到GitHub"
echo "Repository: git@github.com:heimaoqqq/VAE.git"
echo "=" * 50

# 检查是否已经是git仓库
if [ ! -d ".git" ]; then
    echo "📁 初始化Git仓库..."
    git init
    
    echo "🔗 添加远程仓库..."
    git remote add origin git@github.com:heimaoqqq/VAE.git
else
    echo "✅ Git仓库已存在"
    
    # 检查远程仓库
    if git remote get-url origin > /dev/null 2>&1; then
        current_remote=$(git remote get-url origin)
        echo "📍 当前远程仓库: $current_remote"
        
        if [ "$current_remote" != "git@github.com:heimaoqqq/VAE.git" ]; then
            echo "🔄 更新远程仓库地址..."
            git remote set-url origin git@github.com:heimaoqqq/VAE.git
        fi
    else
        echo "🔗 添加远程仓库..."
        git remote add origin git@github.com:heimaoqqq/VAE.git
    fi
fi

# 检查SSH连接
echo "🔐 测试SSH连接..."
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "✅ SSH连接成功"
else
    echo "❌ SSH连接失败，请检查SSH密钥配置"
    echo "参考: https://docs.github.com/en/authentication/connecting-to-github-with-ssh"
    exit 1
fi

# 添加所有文件
echo "📦 添加项目文件..."
git add .

# 检查状态
echo "📋 Git状态:"
git status --short

# 提交
echo "💾 提交更改..."
commit_message="Initial commit: Micro-Doppler VAE Data Augmentation Project

Features:
- VAE training for micro-Doppler spectrograms
- Conditional diffusion model with user ID conditioning
- Kaggle environment optimization
- Comprehensive training and inference scripts
- Evaluation metrics for generation quality

Dataset structure: ID_1/ to ID_31/ (31 users)
Optimized for: 256x256 micro-Doppler time-frequency images"

git commit -m "$commit_message"

# 推送到GitHub
echo "🚀 推送到GitHub..."
if git push -u origin main 2>/dev/null; then
    echo "✅ 推送到main分支成功"
elif git push -u origin master 2>/dev/null; then
    echo "✅ 推送到master分支成功"
else
    echo "🔄 尝试创建main分支并推送..."
    git branch -M main
    git push -u origin main
fi

echo ""
echo "🎉 项目上传完成！"
echo "📍 GitHub地址: https://github.com/heimaoqqq/VAE"
echo ""
echo "📋 下一步 (在Kaggle中):"
echo "1. 创建新的Kaggle Notebook"
echo "2. 克隆仓库: !git clone git@github.com:heimaoqqq/VAE.git"
echo "3. 或使用HTTPS: !git clone https://github.com/heimaoqqq/VAE.git"
echo "4. 进入目录: %cd VAE"
echo "5. 开始训练: !python train_kaggle.py --stage all"
