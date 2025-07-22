#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix final_model: Replace potentially overfitted final model with best model weights
"""

import torch
import shutil
from pathlib import Path
import argparse

def fix_final_model(vqvae_dir):
    """Replace final_model with best_model.pth weights"""
    vqvae_path = Path(vqvae_dir)
    
    # Check file existence
    best_model_path = vqvae_path / "best_model.pth"
    final_model_path = vqvae_path / "final_model"
    
    if not best_model_path.exists():
        print(f"Error: best_model.pth not found: {best_model_path}")
        return False
        
    if not final_model_path.exists():
        print(f"Error: final_model directory not found: {final_model_path}")
        return False
    
    print(f"Checking model files:")
    print(f"   best_model.pth: {best_model_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    safetensors_path = final_model_path / "diffusion_pytorch_model.safetensors"
    if safetensors_path.exists():
        print(f"   final_model: {safetensors_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    try:
        # Load best_model weights
        print(f"\nLoading best model weights...")
        checkpoint = torch.load(best_model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Successfully loaded best model weights (epoch: {epoch})")
        else:
            print(f"Error: incorrect checkpoint format")
            return False
        
        # Backup original final_model
        backup_path = vqvae_path / "final_model_backup"
        if backup_path.exists():
            shutil.rmtree(backup_path)
        
        print(f"Backing up original final_model...")
        shutil.copytree(final_model_path, backup_path)
        print(f"Backup completed: {backup_path}")
        
        # Load model and update with best weights
        print(f"Updating final_model with best weights...")
        
        # Import model class
        import sys
        sys.path.append(str(vqvae_path.parent))
        from models.vqvae_model import MicroDopplerVQVAE
        
        # Load model structure from final_model
        model = MicroDopplerVQVAE.from_pretrained(final_model_path)
        
        # Update with best weights
        model.load_state_dict(model_state)
        
        # Save updated model
        model.save_pretrained(final_model_path)
        
        print(f"Successfully updated final_model with best weights!")
        print(f"Now final_model uses epoch {epoch} best weights")
        
        # Verify file size
        new_size = safetensors_path.stat().st_size / 1024 / 1024
        print(f"Updated file size: {new_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"Update failed: {e}")
        
        # Restore backup
        if backup_path.exists():
            print(f"Restoring original final_model...")
            shutil.rmtree(final_model_path)
            shutil.move(backup_path, final_model_path)
            print(f"Original files restored")
        
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix final_model to avoid overfitting")
    parser.add_argument("--vqvae_dir", type=str, 
                       default="/kaggle/working/outputs/vqvae_transformer/vqvae",
                       help="VQ-VAE output directory")
    
    args = parser.parse_args()
    
    print("Fix final_model tool")
    print("=" * 50)
    print("Goal: Replace potentially overfitted final_model with best model weights")
    print(f"VQ-VAE directory: {args.vqvae_dir}")
    
    success = fix_final_model(args.vqvae_dir)
    
    if success:
        print("\nFix completed!")
        print("Suggestions:")
        print("   1. Run codebook diagnostics to verify model quality")
        print("   2. Start Transformer training")
        print("   3. If issues occur, restore from final_model_backup")
    else:
        print("\nFix failed!")
        print("Suggestions:")
        print("   1. Check if file paths are correct")
        print("   2. Ensure both best_model.pth and final_model exist")
        print("   3. Manually use best_model.pth for subsequent training")

if __name__ == "__main__":
    main()
