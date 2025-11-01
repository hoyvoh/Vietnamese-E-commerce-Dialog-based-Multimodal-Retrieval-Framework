"""
Dialog-based Retriever Module
Contextualized late interaction for Vietnamese e-commerce retrieval
"""

from .model import QwenVLTripletEncoder, triplet_cosine_loss, nce_inbatch_loss
from .dataloader import (
    QwenVLTripletDataset,
    split_wcaptions_dataset,
    triplet_collate_basic,
    triplet_collate_multimodal,
    preprocess_wcaptions,
    infer_category,
    preprocess_text
)
from .train import train_qwen_triplet_retriever, validate_model
from .evaluation import (
    evaluate_search_results,
    evaluate_dialog_success,
    evaluate_modality_accuracy,
    calculate_mrr,
    calculate_recall,
    calculate_ndcg,
    print_metrics
)

__all__ = [
    # Model
    'QwenVLTripletEncoder',
    'triplet_cosine_loss',
    'nce_inbatch_loss',
    # Data loading
    'QwenVLTripletDataset',
    'split_wcaptions_dataset',
    'triplet_collate_basic',
    'triplet_collate_multimodal',
    'preprocess_wcaptions',
    'infer_category',
    'preprocess_text',
    # Training
    'train_qwen_triplet_retriever',
    'validate_model',
    # Evaluation
    'evaluate_search_results',
    'evaluate_dialog_success',
    'evaluate_modality_accuracy',
    'calculate_mrr',
    'calculate_recall',
    'calculate_ndcg',
    'print_metrics'
]
