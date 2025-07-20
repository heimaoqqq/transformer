# Git操作最佳实践指南

## 🚨 常见问题和解决方案

### 1. 进程冲突问题
```bash
❌ 错误: Cannot launch another waiting process while another waiting process is running
```

**原因**: 多个git命令同时执行
**解决方案**:
```bash
# 1. 检查运行中的进程
ps aux | grep git

# 2. 杀死阻塞的git进程
pkill -f "git"

# 3. 等待几秒后重试
sleep 3
git status
```

### 2. Git进程阻塞
```bash
❌ 问题: git commit一直running状态
```

**原因**: commit消息过长或编辑器问题
**解决方案**:
```bash
# 1. 使用简短的commit消息
git commit -m "简短描述"

# 2. 避免使用特殊字符
git commit -m "fix: update environment script"

# 3. 设置Git编辑器
git config --global core.editor "nano"
```

### 3. 网络连接问题
```bash
❌ 问题: git push返回-1错误码
```

**原因**: SSH密钥或网络问题
**解决方案**:
```bash
# 1. 测试SSH连接
ssh -T git@github.com

# 2. 检查远程仓库配置
git remote -v

# 3. 使用HTTPS替代SSH (临时)
git remote set-url origin https://github.com/username/repo.git
```

## 🛠️ 推荐的Git工作流程

### 方法1: 使用辅助脚本
```bash
# 使用我们提供的git_helper.py
python git_helper.py "你的提交消息"
```

### 方法2: 手动分步操作
```bash
# 1. 检查状态 (超时5秒)
timeout 5 git status

# 2. 添加文件 (超时10秒)
timeout 10 git add .

# 3. 提交更改 (简短消息)
git commit -m "fix: update script"

# 4. 推送 (超时30秒)
timeout 30 git push origin main
```

### 方法3: 批处理脚本
```bash
#!/bin/bash
# save as: quick_git.sh

set -e  # 遇到错误立即退出

echo "🔄 Git快速提交脚本"

# 检查参数
if [ $# -eq 0 ]; then
    echo "使用方法: ./quick_git.sh '提交消息'"
    exit 1
fi

COMMIT_MSG="$1"

# 执行Git操作
echo "📁 添加文件..."
git add . || exit 1

echo "💾 提交更改..."
git commit -m "$COMMIT_MSG" || exit 1

echo "🚀 推送到GitHub..."
git push origin main || exit 1

echo "✅ 完成!"
```

## 🔧 Git配置优化

### 1. 设置超时时间
```bash
# 设置HTTP超时
git config --global http.timeout 30

# 设置推送超时
git config --global push.timeout 30
```

### 2. 优化网络配置
```bash
# 使用更快的协议
git config --global protocol.version 2

# 启用压缩
git config --global core.compression 9
```

### 3. 避免编辑器问题
```bash
# 设置简单的编辑器
git config --global core.editor "nano"

# 或者禁用编辑器提示
git config --global advice.detachedHead false
```

## 📋 故障排除检查清单

### 推送前检查
- [ ] 网络连接正常
- [ ] SSH密钥配置正确
- [ ] 没有其他Git进程运行
- [ ] 工作区状态干净
- [ ] 远程仓库可访问

### 推送失败后检查
- [ ] 检查错误消息
- [ ] 验证远程仓库URL
- [ ] 测试SSH连接
- [ ] 检查本地Git配置
- [ ] 尝试fetch远程更新

### 紧急恢复方案
```bash
# 1. 重置到上一次成功的提交
git reset --hard HEAD~1

# 2. 强制推送 (谨慎使用)
git push --force-with-lease origin main

# 3. 创建新分支推送
git checkout -b backup-branch
git push origin backup-branch
```

## 🎯 最佳实践总结

### DO (推荐做法)
✅ 使用简短的commit消息 (<100字符)
✅ 设置合理的超时时间
✅ 分步执行Git操作
✅ 定期检查Git状态
✅ 使用辅助脚本自动化

### DON'T (避免做法)
❌ 同时运行多个Git命令
❌ 使用过长的commit消息
❌ 忽略错误消息
❌ 在网络不稳定时强制推送
❌ 跳过状态检查

## 🚀 推荐工具

### 1. Git辅助脚本
```bash
# 使用项目中的git_helper.py
python vqvae_transformer/git_helper.py "你的提交消息"
```

### 2. Git别名
```bash
# 设置有用的Git别名
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
```

### 3. 监控脚本
```bash
# 检查Git进程
alias gitps='ps aux | grep git'

# 快速状态检查
alias gits='git status --porcelain'
```

---

**记住**: 稳定性比速度更重要。宁可多花几秒钟检查，也不要因为急躁导致操作失败。
