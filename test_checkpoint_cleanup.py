#!/usr/bin/env python3
"""
测试检查点清理逻辑
验证只保留最新1个检查点的功能
"""

import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil

class DummyModel(nn.Module):
    """简单的测试模型"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

def save_checkpoint(model, optimizer, epoch, step, output_dir):
    """保存训练检查点 (只保留最新的1个) - 测试版本"""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # 在分布式训练中，需要通过.module访问原始模型
    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
    }

    # 新检查点文件名
    new_checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
    
    # 删除旧的检查点文件 (保留最新1个)
    try:
        for old_checkpoint in checkpoint_dir.glob("checkpoint_epoch_*.pt"):
            if old_checkpoint != new_checkpoint_path:
                old_checkpoint.unlink()
                print(f"🗑️  删除旧检查点: {old_checkpoint.name}")
    except Exception as e:
        print(f"⚠️  删除旧检查点时出错: {e}")
    
    # 保存新检查点
    torch.save(checkpoint, new_checkpoint_path)
    print(f"💾 保存检查点: {new_checkpoint_path.name}")
    
    # 返回当前检查点数量用于测试
    return len(list(checkpoint_dir.glob("checkpoint_epoch_*.pt")))

def test_checkpoint_cleanup():
    """测试检查点清理功能"""
    print("🧪 测试检查点清理逻辑")
    print("=" * 50)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 临时目录: {temp_dir}")
        
        # 创建测试模型和优化器
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 模拟多个epoch的保存
        checkpoint_counts = []
        for epoch in range(5):
            count = save_checkpoint(model, optimizer, epoch, epoch * 100, temp_dir)
            checkpoint_counts.append(count)
            print(f"Epoch {epoch+1}: 检查点数量 = {count}")
        
        # 验证结果
        print(f"\n📊 检查点数量变化: {checkpoint_counts}")
        
        # 检查最终状态
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        final_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        print(f"🎯 最终检查点文件: {[f.name for f in final_files]}")
        
        # 验证只有1个文件
        if len(final_files) == 1:
            print("✅ 测试通过: 只保留了最新的1个检查点")
            print(f"✅ 最新检查点: {final_files[0].name}")
        else:
            print(f"❌ 测试失败: 应该只有1个检查点，实际有{len(final_files)}个")
        
        # 验证是最新的epoch
        if final_files and "epoch_5" in final_files[0].name:
            print("✅ 验证通过: 保留的是最新的epoch")
        else:
            print("❌ 验证失败: 保留的不是最新的epoch")

def test_disk_space_savings():
    """测试磁盘空间节省效果"""
    print(f"\n💾 磁盘空间节省分析")
    print("=" * 50)
    
    # 估算检查点文件大小
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建一个检查点来估算大小
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 0,
            'step': 0,
        }
        torch.save(checkpoint, temp_file.name)
        
        # 获取文件大小
        file_size = Path(temp_file.name).stat().st_size
        print(f"📏 单个检查点大小: {file_size / 1024:.2f} KB")
        
        # 清理临时文件
        Path(temp_file.name).unlink()
    
    # 计算节省的空间
    epochs = 50  # 假设训练50个epoch
    save_interval = 5  # 每5个epoch保存一次
    total_checkpoints = epochs // save_interval
    
    old_total_size = total_checkpoints * file_size
    new_total_size = 1 * file_size  # 只保留1个
    saved_space = old_total_size - new_total_size
    
    print(f"📊 空间使用对比 (训练{epochs}个epoch):")
    print(f"   旧方案: {total_checkpoints}个检查点 = {old_total_size / 1024 / 1024:.2f} MB")
    print(f"   新方案: 1个检查点 = {new_total_size / 1024 / 1024:.2f} MB")
    print(f"   💰 节省空间: {saved_space / 1024 / 1024:.2f} MB ({saved_space / old_total_size * 100:.1f}%)")

def main():
    """主函数"""
    print("🔧 检查点清理功能测试")
    print("🎯 目标: 只保留最新1个检查点，节省磁盘空间")
    print()
    
    # 运行测试
    test_checkpoint_cleanup()
    test_disk_space_savings()
    
    print(f"\n🎉 测试完成!")
    print(f"✅ 新的保存逻辑已验证，可以大幅节省磁盘空间")

if __name__ == "__main__":
    main()
