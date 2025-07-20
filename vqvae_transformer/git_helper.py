#!/usr/bin/env python3
"""
Git操作辅助脚本
避免Git进程冲突和上传失败问题
"""

import subprocess
import time
import os

def run_git_command(cmd, description="", timeout=30):
    """运行Git命令，避免进程冲突"""
    print(f"🔄 {description}")
    print(f"   命令: {cmd}")
    
    try:
        # 使用较短的超时时间，避免长时间阻塞
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            if result.stdout.strip():
                print(f"   输出: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} 失败")
            if result.stderr.strip():
                print(f"   错误: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时")
        return False
    except Exception as e:
        print(f"❌ {description} 异常: {e}")
        return False

def check_git_status():
    """检查Git状态"""
    print("🔍 检查Git状态...")
    
    # 检查是否在Git仓库中
    if not run_git_command("git rev-parse --git-dir", "检查Git仓库", timeout=5):
        print("❌ 不在Git仓库中")
        return False
    
    # 检查工作区状态
    run_git_command("git status --porcelain", "检查工作区状态", timeout=10)
    
    # 检查远程连接
    if run_git_command("git remote -v", "检查远程仓库", timeout=10):
        return True
    else:
        print("❌ 远程仓库配置有问题")
        return False

def safe_git_add():
    """安全的git add操作"""
    print("\n📁 添加文件到暂存区...")
    
    # 先检查有哪些文件需要添加
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        print("📋 待添加的文件:")
        for line in result.stdout.strip().split('\n'):
            print(f"   {line}")
    
    return run_git_command("git add .", "添加所有文件", timeout=15)

def safe_git_commit(message):
    """安全的git commit操作"""
    print(f"\n💾 提交更改...")
    
    # 使用-m参数避免打开编辑器
    # 限制commit消息长度避免问题
    short_message = message[:100] + "..." if len(message) > 100 else message
    
    cmd = f'git commit -m "{short_message}"'
    return run_git_command(cmd, "提交更改", timeout=20)

def safe_git_push():
    """安全的git push操作"""
    print(f"\n🚀 推送到远程仓库...")
    
    # 先尝试短超时的push
    if run_git_command("git push origin main", "推送到GitHub", timeout=30):
        return True
    
    # 如果失败，尝试强制推送
    print("⚠️ 普通推送失败，尝试其他方法...")
    
    # 检查是否需要pull
    if run_git_command("git fetch origin", "获取远程更新", timeout=15):
        # 检查是否有冲突
        result = subprocess.run("git status", shell=True, capture_output=True, text=True)
        if "behind" in result.stdout:
            print("🔄 远程有更新，尝试pull...")
            if run_git_command("git pull origin main", "拉取远程更新", timeout=20):
                return run_git_command("git push origin main", "重新推送", timeout=30)
    
    return False

def complete_git_workflow(commit_message):
    """完整的Git工作流程"""
    print("🎯 开始Git工作流程")
    print("=" * 50)
    
    # 步骤1: 检查Git状态
    if not check_git_status():
        print("❌ Git状态检查失败")
        return False
    
    # 步骤2: 添加文件
    if not safe_git_add():
        print("❌ 文件添加失败")
        return False
    
    # 步骤3: 提交更改
    if not safe_git_commit(commit_message):
        print("❌ 提交失败")
        return False
    
    # 步骤4: 推送到远程
    if not safe_git_push():
        print("❌ 推送失败")
        return False
    
    print("\n🎉 Git工作流程完成!")
    return True

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python git_helper.py '提交消息'")
        print("示例: python git_helper.py '修复环境安装脚本'")
        return
    
    commit_message = sys.argv[1]
    
    print("🔧 Git操作辅助工具")
    print("避免进程冲突和上传失败")
    print("=" * 40)
    
    success = complete_git_workflow(commit_message)
    
    if success:
        print("\n✅ 所有操作成功完成")
        print("🔗 检查GitHub仓库确认更新")
    else:
        print("\n❌ 操作失败")
        print("💡 建议:")
        print("1. 检查网络连接")
        print("2. 检查SSH密钥配置")
        print("3. 手动执行git命令")

if __name__ == "__main__":
    main()
