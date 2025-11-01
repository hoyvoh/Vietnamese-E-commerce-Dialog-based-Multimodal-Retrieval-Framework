# Vietnamese E-commerce Dialog-based Multimodal Retrieval Framework

**Reference Implementation for Research Paper**: _"A Dialogue-based Multimodal Retrieval Framework for Vietnamese E-commerce RAG System"_

⚠️ **Important Notice**: This codebase has been refactored from original Google Colab notebooks for reference purposes. While the architecture and implementation follow the paper specifications, it may require additional setup and debugging to run successfully. Use this primarily as a reference for understanding the methodology, functions, classes, and system architecture.

## 📄 Paper Overview

This repository implements a complete 3-module architecture for Vietnamese e-commerce product retrieval using dialog-based interactions:

1. **Attribute Predictor** - Multi-label prediction on product attributes
2. **Product Captioner** - Generates comparative captions between products
3. **Dialog-based Retriever** - Contextualized late interaction for token-level matching

### Key Contributions

- **Contextualized Late Interaction**: Token-level fidelity preservation with modality-aware scoring
- **Two-stage Training**: Warm-up + fine-tuning for stable multimodal alignment
- **Vietnamese E-commerce Focus**: Handles code-mixed text, missing diacritics, inconsistent taxonomies
- **Production-ready Pipeline**: Single query-side pass without costly reranking

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Attribute       │    │ Product         │    │ Dialog-based    │
│ Predictor       │───▶│ Captioner       │───▶│ Retriever       │
│                 │    │                 │    │                 │
│ EfficientNet/   │    │ Comparative     │    │ Late Interaction│
│ Swin + Linear   │    │ Caption Gen     │    │ + Qwen2-VL-2B   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
├── main.py                          # Complete training pipeline
├── requirements.txt                 # Dependencies
├── README.md                       # This file
│
├── attribute_predictor/            # Module 1: Attribute Prediction
│   ├── __init__.py                 # Module exports
│   ├── dataset.py                  # ProductImageDataset class
│   └── train.py                    # Training with early stopping
│
├── captioner/                      # Module 2: Product Captioner
│   ├── __init__.py                 # Module exports
│   ├── model.py                    # ProductCaptioningModel
│   ├── dataset.py                  # CaptioningDataset
│   ├── train.py                    # Training with nucleus sampling
│   ├── evaluation.py               # BLEU/ROUGE/CIDEr metrics
│   ├── data_preparation.py         # Complete data prep pipeline
│   ├── prepare_data.py            # Original tokenization (reference)
│   └── prepare_image_pairs.py     # Original pair generation (reference)
│
├── retriever/                      # Module 3: Dialog-based Retriever
│   ├── __init__.py                 # Module exports
│   ├── model.py                    # QwenVLTripletEncoder + Late Interaction
│   ├── dataloader.py              # Triplet data loading
│   ├── train.py                    # Two-stage training pipeline
│   └── evaluation.py              # MRR/Recall@k/Dialog metrics
│
└── data/                          # Dataset (to be uploaded)
    ├── attributes.json            # Product attribute labels
    ├── captioner_pairs.json      # Product pairs for captioning
    ├── wcaptions.json            # Dialog captions data
    └── images/                   # Product images
```

## 🔧 Implementation Details

### Attribute Predictor

- **Backbones**: EfficientNet-B0/B4, Swin Transformer
- **Task**: Multi-label classification on Vietnamese product attributes
- **Evaluation**: Precision/Recall/F1 (macro-averaged)

### Product Captioner

- **Architecture**: Visual features + Attribute vectors → LSTM decoder
- **Training**: Teacher forcing + attribute consistency loss
- **Output**: Short Vietnamese comparative captions (≤20 words)

### Dialog-based Retriever

- **Base Model**: Qwen2-VL-2B-Instruct
- **Late Interaction**: Token-level max-similarity scoring
- **Training**: Two-stage (warm-up → multi-turn fine-tuning)
- **Modes**: Pooled embeddings vs. Late interaction comparison

## 📊 Expected Performance

Based on paper results (Table V - Dialog-based retrieval performance with EfficientNet B0 as captioner backbone):

**Offline Retrieval Metrics**:

- **MRR**: 0.663
- **Recall@1**: 0.475
- **Recall@5**: 0.915
- **nDCG@10**: 0.743

**Online Multi-turn Success**:

- **Dialog@≤1**: 0.35 (35% success within 1 turn)
- **Dialog@≤3**: 0.45 (45% success within 3 turns)
- **Dialog@≤5**: 0.65 (65% success within 5 turns)
- **Mean Turns**: 3.41

_Model: Contextualized Retriever with Late Interaction_

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage Examples

**Complete Pipeline**:

```bash
python main.py --stage all --output_dir outputs/
```

**Individual Stages**:

```bash
# Stage 1: Attribute Predictor
python main.py --stage attribute

# Stage 2: Product Captioner
python main.py --stage captioner

# Stage 3: Dialog Retriever
python main.py --stage retriever

# Stage 4: Evaluation
python main.py --stage evaluation
```

**Custom Configuration**:

```bash
python main.py --config config.json --debug
```

### Configuration

The `create_default_config()` function in `main.py` provides all configurable parameters:

```python
config = {
    "attribute_backbones": ["efficientnet-b0", "efficientnet-b4", "swin"],
    "qwen_model_name": "Qwen/Qwen2-VL-2B-Instruct",
    "enable_late_interaction": True,
    "token_dim": 128,
    "late_interaction_mode": "context",  # or "modality_wise"
    # ... more parameters
}
```

## 📚 Key Classes and Functions

### Attribute Predictor

```python
from attribute_predictor import ProductImageDataset, train_with_early_stopping

# Dataset with image loading and caching
dataset = ProductImageDataset(df, cache_dir="./cache")

# Training with multiple backbones
model = train_with_early_stopping(model, train_loader, val_loader, ...)
```

### Product Captioner

```python
from captioner import ProductCaptioningModel, CaptioningDataset

# Comparative captioning model
model = ProductCaptioningModel(
    backbone="efficientnet-b4",
    vocab_size=736,
    attr_vocab_size=1355
)

# Training with nucleus sampling
train_product_captioning_model(model, train_loader, val_loader, ...)
```

### Dialog-based Retriever

```python
from retriever import QwenVLTripletEncoder, split_wcaptions_dataset

# Late interaction encoder
model = QwenVLTripletEncoder(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    enable_late_interaction=True,
    token_dim=128
)

# Two-stage training
train_qwen_triplet_retriever(model, train_loader, val_loader, ...)
```

## 📊 Data Format

### Attributes Data (`attributes.json`)

```json
{
  "product_id": "12345",
  "image_url": "path/to/image.jpg",
  "attributes": ["red", "A-line", "cotton", "short-sleeve"],
  "title": "Váy A-line màu đỏ tay ngắn"
}
```

### Captions Data (`wcaptions.json`)

```json
{
  "Ir_path": "path/to/reference.jpg",
  "It_path": "path/to/target.jpg",
  "caption": "như cái này nhưng màu xanh và tay dài hơn",
  "Ir_attributes": ["red", "short-sleeve"],
  "It_attributes": ["blue", "long-sleeve"]
}
```

## ⚠️ Known Limitations

1. **Dataset Dependency**: Requires Vietnamese e-commerce product data (being processed)
2. **Model Compatibility**: May need adjustments for different PyTorch/Transformers versions
3. **GPU Requirements**: Qwen2-VL-2B requires significant GPU memory
4. **Language Models**: Some Vietnamese text processing may need fine-tuning

## 🔬 Research Use

This codebase is designed for:

- **Understanding** the paper's methodology and architecture
- **Reproducing** experiments with proper dataset
- **Extending** to other languages or domains
- **Benchmarking** against the proposed approach

## 📖 Citation

If you use this code for research, please cite the original paper:

```bibtex
@article{vietnamese_ecommerce_rag_2024,
  title={A Dialogue-based Multimodal Retrieval Framework for Vietnamese E-commerce RAG System},
  author={Ho Ngoc Tuong Vy, Ngo Thuan Phat, Nguyen Huynh Minh Huy, Nguyen Minh Nhut, Nguyen Dinh Thuan},
  year={2024},
  institution={University of Information Technology, VNU-HCM}
}
```

## 📄 License

This research code is provided for academic and research purposes. Please check with the original authors for commercial usage rights.

## 🔄 Updates

- **Dataset**: Currently being processed and will be uploaded soon
- **Documentation**: Additional tutorials and examples coming
- **Compatibility**: Testing with latest library versions in progress

---

**Note**: This is a reference implementation refactored from research notebooks. For production use, additional engineering and testing would be required.
