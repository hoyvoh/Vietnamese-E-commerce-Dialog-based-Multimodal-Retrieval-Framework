"""
Product Captioner Module
Comparative caption generation for Vietnamese e-commerce products
"""

from .dataset import CaptioningDataset
from .model import (
    AttributePredictor, 
    ProductCaptioningModel,
    VisualEmbedding,
    AttributeEmbedding,
    JointEncoding,
    PositionalEncoder
)
from .train import (
    train_product_captioning_model,
    nucleus_sampling_vectorized,
    pretrain_decoder,
    decode_batch_ids,
    compute_rouge_l,
    compute_attr_accuracy
)
from .evaluation import evaluate_model_on_testset
from .prepare_data import load_tokenizer_and_vocab, retokenize_captions
from .prepare_image_pairs import (
    find_product_pairs,
    CaptionPair,
    generate_comparative_captions_batch,
    save_checkpoint,
    load_checkpoint
)

__all__ = [
    # Dataset
    'CaptioningDataset',
    # Models
    'AttributePredictor',
    'ProductCaptioningModel', 
    'VisualEmbedding',
    'AttributeEmbedding',
    'JointEncoding',
    'PositionalEncoder',
    # Training
    'train_product_captioning_model',
    'nucleus_sampling_vectorized',
    'pretrain_decoder',
    'decode_batch_ids',
    'compute_rouge_l',
    'compute_attr_accuracy',
    # Evaluation
    'evaluate_model_on_testset',
    # Data preparation
    'load_tokenizer_and_vocab',
    'retokenize_captions',
    'find_product_pairs',
    'CaptionPair',
    'generate_comparative_captions_batch',
    'save_checkpoint',
    'load_checkpoint'
]
