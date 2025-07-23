"""
VQ-VAE + Transformer 模型包
包含防坍缩VQ-VAE和条件Transformer模型
"""

from .vqvae_model import MicroDopplerVQVAE, EMAVectorQuantizer
from .transformer_model import MicroDopplerTransformer, UserConditionEncoder

__all__ = [
    'MicroDopplerVQVAE',
    'EMAVectorQuantizer', 
    'MicroDopplerTransformer',
    'UserConditionEncoder',
]
