import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertModel, DistilBertTokenizer, GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from torch.amp import GradScaler, autocast
import os
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.spice.spice import Spice

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torchvision.models import efficientnet_b0, efficientnet_b4, swin_v2_b
import re

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=50):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

import torch
import torch.nn as nn

class JointEncoding(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        concat_dim = input_dim * 4  # ref_embed, tgt_embed, diff, prod
        self.linear1 = nn.Linear(concat_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, ref_embed: torch.Tensor, tgt_embed: torch.Tensor) -> torch.Tensor:
        diff = ref_embed - tgt_embed
        prod = ref_embed * tgt_embed
        concat_feat = torch.cat([ref_embed, tgt_embed, diff, prod], dim=-1)
        x = self.linear1(concat_feat)
        x = self.relu(x)
        x = self.dropout_layer(x)
        x = self.linear2(x)
        x = self.norm(x)
        return x

class VisualEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        x = self.projection(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.norm(x)
        return x

class AttributeEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.mlp(x)

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b4, swin_v2_b
from torchvision.models.efficientnet import EfficientNet_B0_Weights, EfficientNet_B4_Weights
from torchvision.models.swin_transformer import Swin_V2_B_Weights

class AttributePredictor(nn.Module):
    """
    A wrapper around common vision backbones for predicting attributes and
    optionally returning intermediate features. The backbone can be loaded
    from a pretrained checkpoint or base model library. When use_base=True,
    classifier_dim sets the number of output labels for the classifier,
    overriding num_labels.
    """

    def __init__(
        self,
        model_name: str,
        pretrained_backbone_path: str = None,
        num_labels: int = 300,
        freeze_backbone: bool = True,
        use_base: bool = False,
        classifier_dim: int = None
    ) -> None:
        super().__init__()
        self.model_name = model_name.lower()
        self.num_labels = num_labels
        self.freeze_backbone = freeze_backbone
        self.use_base = use_base
        self.classifier_dim = classifier_dim

        # Validate usage conditions
        if self.use_base and pretrained_backbone_path is not None:
            raise ValueError("Cannot use both use_base=True and pretrained_backbone_path. Set pretrained_backbone_path=None when use_base=True.")
        if self.use_base and classifier_dim is not None and (not isinstance(classifier_dim, int) or classifier_dim <= 0):
            raise ValueError(f"classifier_dim must be a positive integer when use_base=True, got {classifier_dim}")
        if self.model_name not in ["efficientnetb0", "efficientnetb4", "swinbased"]:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        try:
            self.model, self.feature_dim = self._load_and_configure_model(pretrained_backbone_path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")

        # Optionally freeze backbone parameters
        if self.freeze_backbone:
            self._freeze_parameters(self.model)

    def _freeze_parameters(self, model: nn.Module) -> None:
        """Disable gradients for all parameters in the given model."""
        for param in model.parameters():
            param.requires_grad = False

    def _load_and_configure_model(self, pretrained_backbone_path):
        """
        Load a vision backbone and configure its classifier head. If use_base=True,
        load from torchvision's pretrained weights and use classifier_dim (if provided)
        as the classifier output dimension. Otherwise, use num_labels and load from
        pretrained_backbone_path or no weights.
        """
        if self.model_name == "efficientnetb0":
            if self.use_base:
                model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
                in_features = model.classifier[1].in_features  # 1280
            else:
                model = efficientnet_b0(weights=None)
                in_features = model.classifier[1].in_features
        elif self.model_name == "efficientnetb4":
            if self.use_base:
                model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
                in_features = model.classifier[1].in_features  # 1792
            else:
                model = efficientnet_b4(weights=None)
                in_features = model.classifier[1].in_features
        elif self.model_name == "swinbased":
            if self.use_base:
                model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
                in_features = model.head.in_features  # 1024
            else:
                model = swin_v2_b(weights=None)
                in_features = model.head.in_features
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        # Set classifier output dimension
        output_dim = self.classifier_dim if self.use_base and self.classifier_dim is not None else self.num_labels
        if self.model_name in ["efficientnetb0", "efficientnetb4"]:
            model.classifier[1] = nn.Linear(in_features, output_dim)
            nn.init.xavier_uniform_(model.classifier[1].weight)
            nn.init.zeros_(model.classifier[1].bias)
        elif self.model_name == "swinbased":
            model.head = nn.Linear(in_features, output_dim)
            nn.init.xavier_uniform_(model.head.weight)
            nn.init.zeros_(model.head.bias)

        # Load external weights if provided and use_base=False
        if not self.use_base and pretrained_backbone_path:
            try:
                loaded_model = torch.load(pretrained_backbone_path, map_location='cpu')
                if isinstance(loaded_model, dict) and 'state_dict' in loaded_model:
                    model.load_state_dict(loaded_model['state_dict'], strict=False)
                elif hasattr(loaded_model, 'state_dict'):
                    model.load_state_dict(loaded_model.state_dict(), strict=False)
                else:
                    model.load_state_dict(loaded_model, strict=False)
            except Exception as e:
                print(f"Error loading pretrained weights from {pretrained_backbone_path}: {e}")
                # Reinitialize classifier/head if loading fails
                if self.model_name in ["efficientnetb0", "efficientnetb4"]:
                    nn.init.xavier_uniform_(model.classifier[1].weight)
                    nn.init.zeros_(model.classifier[1].bias)
                elif self.model_name == "swinbased":
                    nn.init.xavier_uniform_(model.head.weight)
                    nn.init.zeros_(model.head.bias)

        return model, in_features

    def unfreeze_backbone(self, partial: bool = True) -> None:
        """
        Unfreeze parameters for training. If partial=True, only the last few
        layers of the vision backbone are unfrozen. If False, the entire backbone
        is unfrozen.
        """
        if partial:
            for name, param in self.model.named_parameters():
                if 'features.8' in name or 'features.7' in name or 'layers.3' in name:
                    param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass through the backbone. Returns attribute predictions and
        optionally intermediate feature vectors if return_features=True.
        """
        if return_features:
            if self.model_name in ["efficientnetb0", "efficientnetb4"]:
                features = self.model.features(x)
                features = self.model.avgpool(features).view(features.size(0), -1)
                attributes = self.model.classifier(features)
                return attributes, features
            elif self.model_name == "swinbased":
                features = self.model.features(x)
                features = self.model.norm(features)
                features = self.model.permute(features)
                features = self.model.avgpool(features).view(features.size(0), -1)
                attributes = self.model.head(features)
                return attributes, features
        return self.model(x)
    
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class ProductCaptioningModel(nn.Module):
    """
    A multimodal model for product captioning. Combines visual features from two
    product images with learned attribute embeddings and passes the joint embedding
    through an LSTM decoder to produce captions. Supports AttributePredictor
    with use_base and classifier_dim for custom classifier output dimensions.
    """
    def __init__(
        self,
        token_to_id: dict,
        id_to_token: dict,
        model_name: str = "efficientnet_b0",
        pretrained_backbone_path: str = None,
        num_labels: int = 300,
        d_model: int = 512,
        lstm_hidden_dim: int = 512,
        lstm_layers: int = 2,
        seq_len: int = 35,
        dropout: float = 0.1,
        use_cross_attention: bool = False,
        freeze_backbone: bool = True,
        use_base: bool = False,
        classifier_dim: int = None
    ) -> None:
        super().__init__()
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.vocab_size = len(token_to_id)
        self.pad_token_id = token_to_id['<pad>']
        self.bos_token_id = token_to_id['<s>']
        self.eos_token_id = token_to_id['</s>']
        self.d_model = d_model
        self.use_cross_attention = use_cross_attention
        self.pretrained_backbone_path = pretrained_backbone_path
        self.use_base = use_base
        self.classifier_dim = classifier_dim

        # Validate usage conditions
        if self.use_base and pretrained_backbone_path is not None:
            raise ValueError("Cannot use both use_base=True and pretrained_backbone_path. Set pretrained_backbone_path=None when use_base=True.")
        if self.use_base and classifier_dim is not None and (not isinstance(classifier_dim, int) or classifier_dim <= 0):
            raise ValueError(f"classifier_dim must be a positive integer when use_base=True, got {classifier_dim}")

        # Attribute predictor
        self.attribute_predictor = AttributePredictor(
            model_name=model_name,
            pretrained_backbone_path=pretrained_backbone_path,
            num_labels=num_labels,
            freeze_backbone=freeze_backbone,
            use_base=use_base,
            classifier_dim=classifier_dim
        )
        self.feature_dim = self.attribute_predictor.feature_dim
        self.attr_output_dim = classifier_dim if use_base and classifier_dim is not None else num_labels

        # Embeddings for visual features and attributes
        self.visual_embedding = VisualEmbedding(
            input_dim=self.feature_dim,
            d_model=d_model,
            dropout=dropout
        )
        self.attribute_embedding = AttributeEmbedding(
            input_dim=self.attr_output_dim,
            d_model=d_model,
            dropout=dropout
        )

        # Joint encoding
        self.joint_encoding = JointEncoding(
            input_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout
        )

        # LSTM decoder
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.output_layer = nn.Linear(lstm_hidden_dim, self.vocab_size)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        # Learnable scalar controlling attribute weight
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        Ir: torch.Tensor,
        It: torch.Tensor,
        caption_input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        """Forward pass through the captioning model.

        Args:
            Ir: Reference images [batch_size, 3, H, W].
            It: Target images [batch_size, 3, H, W].
            caption_input_ids: Tokenized captions [batch_size, seq_len].
            attention_mask: Optional mask for captions [batch_size, seq_len].

        Returns:
            Tuple of (logits, attr_r, attr_t), where logits contains token predictions
            for the caption and attr_* are the raw attribute logits for the images.
        """
        # Validate token IDs
        if caption_input_ids.max() >= self.vocab_size or caption_input_ids.min() < 0:
            raise ValueError(
                f"Invalid token IDs: min={caption_input_ids.min()}, max={caption_input_ids.max()}, "
                f"vocab_size={self.vocab_size}"
            )

        # Run attribute predictor and obtain features
        attr_r, Ir_feat = self.attribute_predictor(Ir, return_features=True)
        attr_t, It_feat = self.attribute_predictor(It, return_features=True)

        # Compute embeddings
        Ir_embed = self.visual_embedding(Ir_feat)
        It_embed = self.visual_embedding(It_feat)
        Ar_embed = self.attribute_embedding(attr_r)
        At_embed = self.attribute_embedding(attr_t)

        # Apply learnable scaling to attribute embeddings
        ref_embed = Ir_embed + self.alpha * Ar_embed
        tgt_embed = It_embed + self.alpha * At_embed
        joint_embed = self.joint_encoding(ref_embed, tgt_embed)

        # Token embeddings
        token_embeds = self.token_embedding(caption_input_ids)

        # Initialize LSTM hidden state with joint embedding
        batch_size = Ir.size(0)
        h_0 = joint_embed.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=joint_embed.device)
        lstm_output, _ = self.lstm(token_embeds, (h_0, c_0))

        # Compute logits
        logits = self.output_layer(lstm_output)
        return logits, attr_r, attr_t
    