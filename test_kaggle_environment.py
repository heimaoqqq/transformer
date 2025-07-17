#!/usr/bin/env python3
"""
Kaggle环境专用测试脚本
在Kaggle Notebook中运行，验证环境和数据集
"""

import os
import sys
from pathlib import Path
import subprocess

def test_kaggle_environment():
    """测试Kaggle环境"""
    print("🔍 检测Kaggle环境...")
    
    # 检查是否在Kaggle中
    if "/kaggle" in os.getcwd():
        print("✅ 检测到Kaggle环境")
        
        # 检查工作目录
        working_dir = Path("/kaggle/working")
        if working_dir.exists():
            print(f"✅ 工作目录存在: {working_dir}")
        
        # 检查输入目录
        input_dir = Path("/kaggle/input")
        if input_dir.exists():
            print(f"✅ 输入目录存在: {input_dir}")
            
            # 列出可用数据集
            datasets = list(input_dir.iterdir())
            print(f"📊 可用数据集: {[d.name for d in datasets]}")
        
        return True
    else:
        print("⚠️  不在Kaggle环境中")
        return False

def test_dataset_structure():
    """测试数据集结构"""
    print("\n📁 检测数据集结构...")
    
    data_path = Path("/kaggle/input/dataset")
    
    if not data_path.exists():
        print(f"❌ 数据集路径不存在: {data_path}")
        
        # 尝试查找其他可能的路径
        input_dir = Path("/kaggle/input")
        if input_dir.exists():
            print("🔍 搜索可能的数据集路径...")
            for item in input_dir.iterdir():
                if item.is_dir():
                    print(f"   发现目录: {item}")
                    
                    # 检查是否包含ID_开头的目录
                    id_dirs = [d for d in item.iterdir() if d.is_dir() and d.name.startswith('ID_')]
                    if id_dirs:
                        print(f"   ✅ 可能的数据集路径: {item}")
                        print(f"   包含 {len(id_dirs)} 个ID_目录")
        
        return False
    
    print(f"✅ 数据集路径存在: {data_path}")
    
    # 检查用户目录
    user_dirs = []
    for item in data_path.iterdir():
        if item.is_dir() and item.name.startswith('ID_'):
            try:
                user_id = int(item.name.split('_')[1])
                user_dirs.append((user_id, item))
            except ValueError:
                print(f"⚠️  无效的用户目录名: {item.name}")
    
    # 按用户ID排序
    user_dirs.sort(key=lambda x: x[0])
    
    print(f"📊 找到 {len(user_dirs)} 个用户目录")
    
    if len(user_dirs) == 0:
        print("❌ 未找到有效的用户目录")
        return False
    
    # 检查前几个用户的图像
    total_images = 0
    for user_id, user_dir in user_dirs[:5]:
        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            images.extend(user_dir.glob(ext))
        
        print(f"   ID_{user_id}: {len(images)} 张图像")
        total_images += len(images)
    
    if len(user_dirs) > 5:
        print(f"   ... 还有 {len(user_dirs) - 5} 个用户目录")
        
        # 统计所有图像
        for user_id, user_dir in user_dirs[5:]:
            images = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                images.extend(user_dir.glob(ext))
            total_images += len(images)
    
    print(f"📈 总计: {total_images} 张图像")
    
    if total_images == 0:
        print("❌ 未找到任何图像文件")
        return False
    
    return True

def test_gpu_availability():
    """测试GPU可用性"""
    print("\n🔥 检测GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"✅ GPU可用: {gpu_name}")
            print(f"   GPU数量: {gpu_count}")
            print(f"   GPU内存: {gpu_memory:.1f} GB")
            
            # 检查内存是否足够
            if gpu_memory >= 8:
                print("✅ GPU内存充足 (>= 8GB)")
            else:
                print("⚠️  GPU内存较少 (< 8GB)，建议减小batch_size")
            
            return True
        else:
            print("❌ GPU不可用")
            return False
            
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def test_disk_space():
    """测试磁盘空间"""
    print("\n💾 检测磁盘空间...")
    
    try:
        import shutil
        
        # 检查工作目录空间
        working_dir = "/kaggle/working"
        if os.path.exists(working_dir):
            total, used, free = shutil.disk_usage(working_dir)
            free_gb = free / 1024**3
            
            print(f"✅ 工作目录可用空间: {free_gb:.1f} GB")
            
            if free_gb >= 5:
                print("✅ 磁盘空间充足 (>= 5GB)")
            else:
                print("⚠️  磁盘空间较少 (< 5GB)，注意清理临时文件")
            
            return True
        else:
            print("⚠️  无法检测工作目录")
            return False
            
    except Exception as e:
        print(f"❌ 磁盘空间检测失败: {e}")
        return False

def install_dependencies():
    """安装依赖"""
    print("\n📦 安装项目依赖...")
    
    try:
        # 安装核心依赖
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 核心依赖安装完成")
        
        # 安装可选的评估工具
        optional_packages = [
            "lpips==0.1.4",
            "pytorch-fid==0.3.0"
        ]
        
        for package in optional_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} 安装成功")
            except:
                print(f"⚠️  {package} 安装失败，跳过")
        
        return True
        
    except Exception as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def run_compatibility_tests():
    """运行兼容性测试"""
    print("\n🧪 运行兼容性测试...")
    
    try:
        # 运行基础依赖测试
        result = subprocess.run([sys.executable, "test_dependencies.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 基础依赖测试通过")
        else:
            print("⚠️  基础依赖测试失败")
            print(result.stdout)
            print(result.stderr)
        
        # 运行Diffusers测试
        result = subprocess.run([sys.executable, "test_diffusers_compatibility.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Diffusers兼容性测试通过")
            return True
        else:
            print("⚠️  Diffusers兼容性测试失败")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        return False

def main():
    """主函数"""
    print("🧪 Kaggle环境完整测试")
    print("=" * 50)
    
    tests = [
        ("Kaggle环境", test_kaggle_environment),
        ("数据集结构", test_dataset_structure),
        ("GPU可用性", test_gpu_availability),
        ("磁盘空间", test_disk_space),
    ]
    
    # 运行基础测试
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # 检查是否需要安装依赖
    if not os.path.exists("requirements.txt"):
        print("\n❌ requirements.txt 不存在")
        print("请确保在项目根目录中运行此脚本")
        return
    
    # 安装依赖
    print(f"\n{'='*20} 安装依赖 {'='*20}")
    install_result = install_dependencies()
    results.append(("依赖安装", install_result))
    
    # 运行兼容性测试
    if install_result:
        print(f"\n{'='*20} 兼容性测试 {'='*20}")
        compat_result = run_compatibility_tests()
        results.append(("兼容性测试", compat_result))
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    passed = 0
    critical_failed = False
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        
        if result:
            passed += 1
        elif test_name in ["Kaggle环境", "数据集结构", "GPU可用性"]:
            critical_failed = True
    
    print(f"\n总计: {passed}/{len(results)} 项测试通过")
    
    if critical_failed:
        print("\n❌ 关键测试失败，无法继续训练")
        print("请检查:")
        print("1. 是否在Kaggle环境中运行")
        print("2. 数据集是否正确添加")
        print("3. GPU是否可用")
    elif passed == len(results):
        print("\n🎉 所有测试通过！")
        print("✅ 环境已准备就绪，可以开始训练")
        print("\n📋 下一步:")
        print("!python train_kaggle.py --stage all")
    else:
        print("\n⚠️  部分测试失败，但可以尝试训练")
        print("建议先解决失败的测试项")

if __name__ == "__main__":
    main()
