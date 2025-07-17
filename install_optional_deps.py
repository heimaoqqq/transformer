#!/usr/bin/env python3
"""
可选依赖安装脚本
根据环境和需求安装额外的包
"""

import subprocess
import sys
import torch

def run_pip_install(package, description=""):
    """安装包"""
    print(f"🔄 安装 {package} - {description}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} 安装失败: {e}")
        return False

def install_evaluation_tools():
    """安装评估工具"""
    print("\n📊 安装评估工具...")
    
    packages = [
        ("lpips==0.1.4", "感知损失计算"),
        ("pytorch-fid==0.3.0", "FID评估指标"),
    ]
    
    for package, desc in packages:
        run_pip_install(package, desc)

def install_memory_optimization():
    """安装内存优化工具"""
    print("\n🚀 安装内存优化工具...")
    
    if torch.cuda.is_available():
        print("检测到CUDA，安装xformers...")
        # xformers需要特定的PyTorch版本
        torch_version = torch.__version__
        print(f"PyTorch版本: {torch_version}")
        
        if "2.1" in torch_version:
            run_pip_install("xformers==0.0.22", "内存优化 (CUDA)")
        else:
            print("⚠️  xformers版本可能不兼容，跳过安装")
    else:
        print("⚠️  未检测到CUDA，跳过xformers安装")

def install_experiment_tracking():
    """安装实验跟踪工具"""
    print("\n📈 安装实验跟踪工具...")
    
    packages = [
        ("wandb==0.16.0", "Weights & Biases实验跟踪"),
        ("tensorboard==2.15.1", "TensorBoard可视化"),
    ]
    
    for package, desc in packages:
        run_pip_install(package, desc)

def install_development_tools():
    """安装开发工具"""
    print("\n🛠️ 安装开发工具...")
    
    packages = [
        ("jupyter>=1.0.0", "Jupyter Notebook"),
        ("ipywidgets>=8.0.0", "交互式组件"),
        ("matplotlib>=3.7.0", "绘图工具"),
    ]
    
    for package, desc in packages:
        run_pip_install(package, desc)

def main():
    """主函数"""
    print("🔧 微多普勒VAE项目 - 可选依赖安装")
    print("=" * 50)
    
    print("选择要安装的可选依赖:")
    print("1. 评估工具 (LPIPS, FID)")
    print("2. 内存优化 (xformers)")
    print("3. 实验跟踪 (wandb, tensorboard)")
    print("4. 开发工具 (jupyter, matplotlib)")
    print("5. 全部安装")
    print("0. 退出")
    
    while True:
        try:
            choice = input("\n请选择 (0-5): ").strip()
            
            if choice == "0":
                print("退出安装")
                break
            elif choice == "1":
                install_evaluation_tools()
            elif choice == "2":
                install_memory_optimization()
            elif choice == "3":
                install_experiment_tracking()
            elif choice == "4":
                install_development_tools()
            elif choice == "5":
                print("安装所有可选依赖...")
                install_evaluation_tools()
                install_memory_optimization()
                install_experiment_tracking()
                install_development_tools()
                break
            else:
                print("无效选择，请重新输入")
                continue
                
            # 询问是否继续
            cont = input("\n是否继续安装其他依赖? (y/n): ").strip().lower()
            if cont != 'y':
                break
                
        except KeyboardInterrupt:
            print("\n\n安装中断")
            break
    
    print("\n✅ 可选依赖安装完成")
    print("💡 提示: 运行 'python test_dependencies.py' 验证安装")

if __name__ == "__main__":
    main()
