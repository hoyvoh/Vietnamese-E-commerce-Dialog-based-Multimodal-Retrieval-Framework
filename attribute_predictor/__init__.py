"""
Attribute Predictor Module
Multi-label attribute prediction for Vietnamese e-commerce products
"""

from .dataset import ProductImageDataset, load_images_and_labels
from .train import train_with_early_stopping

__all__ = [
    'ProductImageDataset',
    'load_images_and_labels',
    'train_with_early_stopping'
]
