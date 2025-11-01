import os
import json
import logging
import math
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.io as tio
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# from qwen_vl_utils import process_vision_info  # ← Tùy chọn, chưa dùng

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _tensor_stats(x: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    try:
        xc = x.detach()
        stats = {
            f"{prefix}shape0": int(xc.shape[0]) if xc.ndim > 0 else 0,
            f"{prefix}mean": float(torch.nan_to_num(xc).float().mean().item()),
            f"{prefix}norm_mean": float(torch.linalg.vector_norm(xc.reshape(xc.shape[0], -1), dim=1).mean().item()) if xc.ndim >= 2 else float(torch.linalg.vector_norm(xc).item()),
            f"{prefix}requires_grad": bool(xc.requires_grad),
        }
        return stats
    except Exception:
        return {f"{prefix}shape0": -1, f"{prefix}mean": 0.0, f"{prefix}norm_mean": 0.0, f"{prefix}requires_grad": False}

def _write_jsonl(path: str, payload: Union[Dict[str, Any], List[Dict[str, Any]]]):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = payload if isinstance(payload, list) else [payload]
        with open(path, "a", encoding="utf-8") as f:
            for row in rows:
                row = dict(row)
                row["cwd"] = os.getcwd()
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"cannot write jsonl to {path}: {e}")

def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _guard_tensor(tensor: torch.Tensor, tag: str, strict: bool = False) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{tag} must be a Tensor, got {type(tensor)}")
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        if strict:
            raise ValueError(f"{tag} contains NaN/Inf")
        logger.warning(f"{tag} contains NaN/Inf -> zeroing")
        return torch.zeros_like(tensor)
    return tensor

def _check_vram_threshold(threshold_ratio: float = 0.9) -> bool:
    if not torch.cuda.is_available():
        return False
    allocated = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
    return allocated > threshold_ratio

def triplet_cosine_loss(query: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    d_pos = 1 - (query * positive).sum(-1)
    d_neg = 1 - (query * negative).sum(-1)
    loss = F.relu(margin + d_pos - d_neg).mean()
    return loss

def nce_inbatch_loss(query: torch.Tensor, keys: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = query @ keys.T / temperature
    targets = torch.arange(query.size(0), device=query.device)
    return F.cross_entropy(logits, targets)

class QwenVLTripletEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        out_dim: int = 1024,
        device: str = "cuda",
        cache_dir: str = "cache",
        revision: str = None,
        debug: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        min_pixels: int = 256*28*28,
        max_pixels: int = 896*28*28,
        dtype=torch.bfloat16,
        strict_numerics: bool = True,
        # === THÊM MỚI: Late Interaction ===
        enable_late_interaction: bool = False,
        token_dim: int = 128,
        max_query_tokens: int = 64,
        max_doc_tokens: int = 256,
        late_interaction_mode: str = "both",  # "both", "modality_wise", "context",
        late_interaction_temperature=1.0
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.debug = debug
        self.strict_numerics = strict_numerics
        self.out_dim = out_dim
        self.dtype = dtype

        # === KHỞI TẠO VLM (giữ nguyên) ===
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.device_count() > 1 else None,
            attn_implementation="sdpa",
            trust_remote_code=True,
            cache_dir=cache_dir,
            revision=revision
        )
        if torch.cuda.device_count() <= 1:
            self.vlm = self.vlm.to(self.device)
        self.vlm.eval()
        self.vlm_processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
            cache_dir=cache_dir,
            revision=revision
        )
        if hasattr(self.vlm_processor, "tokenizer"):
            self.vlm_processor.tokenizer.padding_side = "right"

        tok = self.vlm_processor.tokenizer
        self.vid_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
        self.vid_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
        self.image_token_id = getattr(self.vlm_processor, "image_token_id", None)

        self.hidden_size = self.vlm.config.hidden_size
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, out_dim)
        ).to(self.device)

        # === THÊM: Late Interaction Components ===
        self.enable_late_interaction = enable_late_interaction
        self.token_dim = token_dim
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens
        self.late_interaction_mode = late_interaction_mode

        if self.enable_late_interaction:
            self.token_projection = nn.Sequential(
                nn.Linear(self.hidden_size, token_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(token_dim, token_dim)
            ).to(self.device)

            self.visual_token_proj = nn.Linear(token_dim, token_dim).to(self.device)
            self.text_token_proj = nn.Linear(token_dim, token_dim).to(self.device)
            self.late_interaction_temperature = nn.Parameter(torch.tensor(late_interaction_temperature))

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=16,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.1,
                    bias="none"
                )
                self.vlm = get_peft_model(self.vlm, lora_config)
            except Exception as e:
                logger.warning(f"LoRA failed: {e}, proceeding without LoRA")
                use_lora = False
        self._setup_trainable_params(use_lora)

    def _setup_trainable_params(self, use_lora: bool = False):
        for param in self.vlm.parameters():
            param.requires_grad = False
        for param in self.projection_head.parameters():
            param.requires_grad = True
        self.fusion_alpha.requires_grad = True
        if use_lora:
            for name, param in self.vlm.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _safe_l2_normalize(self, x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        norm = torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)
        norm = torch.clamp(norm, min=eps)
        return x / norm

    def _tensor_to_pil_list(self, tensor: torch.Tensor) -> List[Image.Image]:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError(f"Expected 4D tensor with 3 channels, got shape {tensor.shape}")
        tensor = tensor.clamp(0, 1) * 255.0
        tensor = tensor.to(dtype=torch.uint8).permute(0, 2, 3, 1)
        pil_list = [Image.fromarray(t.detach().cpu().numpy()).convert("RGB") for t in tensor]
        return pil_list

    def _build_chat_messages(self, images: List[Image.Image], texts: List[str], mode: str) -> List[List[Dict]]:
        messages = []
        for img, text in zip(images, texts):
            content = []
            if mode != "text":
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": text or " "})
            messages.append([{"role": "user", "content": content}])
        return messages

    def _apply_chat_template(self, messages_batch: List[List[Dict]]) -> List[str]:
        templated_texts = []
        for messages in messages_batch:
            try:
                templated = self.vlm_processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                templated_texts.append(templated)
            except Exception as e:
                logger.error(f"Failed to apply template: {e}")
                text_content = " ".join([
                    item["text"] for item in messages[0]["content"]
                    if item["type"] == "text"
                ])
                templated_texts.append(text_content or " ")
        return templated_texts

    def _encode_batch(
        self,
        images: List[Image.Image],
        texts: List[str],
        mode: str = "both",
        is_training: bool = False,
        return_tokens: Optional[str] = None  # "pooled", "tokens", "both"
    ) -> Union[torch.Tensor, Dict]:
        """
        Enhanced _encode_batch with optional token-level outputs.
        """
        if return_tokens is None:
            return_tokens = "both" if self.enable_late_interaction else "pooled"

        if mode not in ["both", "image", "text"]:
            raise ValueError(f"Invalid mode: {mode}")
        if not isinstance(images, list) or not all(isinstance(img, Image.Image) for img in images):
            raise ValueError(f"Expected list of PIL images, got {type(images)}")

        pil_images = [img.convert("RGB") for img in images]
        processed_texts = [(text or "").strip() for text in texts]

        try:
            msgs = self._build_chat_messages(pil_images, processed_texts, mode)
            inputs = self.vlm_processor(
                text=self._apply_chat_template(msgs),
                images=pil_images if mode != "text" else None,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        except Exception as e:
            logger.error(f"VLM processor failed: {e}")
            raise

        device = self.device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device=device, non_blocking=True)

        with torch.no_grad() if not is_training else torch.enable_grad():
            try:
                outputs = self.vlm(**inputs, output_hidden_states=True)
            except Exception as e:
                logger.error(f"VLM forward failed: {e}")
                raise

        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]
        else:
            raise ValueError("Cannot find hidden states")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size, seq_len = input_ids.shape
        visual_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        is_image_token = (input_ids == self.image_token_id) if self.image_token_id else torch.zeros_like(input_ids, dtype=torch.bool)

        # Xây dựng visual_mask (giữ nguyên logic cũ)
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            image_grid_thw = inputs["image_grid_thw"]
            for b in range(batch_size):
                pos = is_image_token[b].nonzero(as_tuple=True)[0]
                if pos.numel() > 0 and image_grid_thw[b].numel() >= 3:
                    h, w = image_grid_thw[b][-2:]
                    num_vis = int(h * w)
                    if num_vis > 0:
                        start = pos[0].item()
                        end = min(start + num_vis, seq_len)
                        visual_mask[b, start:end] = 1.0
        else:
            is_start = (input_ids == self.vid_start_id)
            is_end = (input_ids == self.vid_end_id)
            for b in range(batch_size):
                starts = is_start[b].nonzero(as_tuple=True)[0]
                ends = is_end[b].nonzero(as_tuple=True)[0]
                if starts.numel() > 0 and ends.numel() > 0:
                    s, e = starts[0].item(), ends[0].item()
                    if e > s:
                        visual_mask[b, s:e+1] = 1.0

        is_valid_token = attention_mask.bool()
        text_mask = (is_valid_token & ~visual_mask.bool()).float()

        if mode == "image":
            text_mask = torch.zeros_like(text_mask)
        elif mode == "text":
            visual_mask = torch.zeros_like(visual_mask)

        def masked_mean_pool(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            mask_expanded = mask.unsqueeze(-1)
            masked_hidden = hidden_states * mask_expanded
            sum_hidden = masked_hidden.sum(dim=1)
            sum_mask = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return sum_hidden / sum_mask

        # === POOLED EMBEDDINGS (giữ nguyên) ===
        pooled_output = None
        if return_tokens in ["pooled", "both"]:
            visual_repr = masked_mean_pool(hidden_states, visual_mask)
            text_repr = masked_mean_pool(hidden_states, text_mask)
            alpha = torch.sigmoid(self.fusion_alpha)
            fused_repr = alpha * visual_repr + (1.0 - alpha) * text_repr
            embeddings = self.projection_head(fused_repr)
            pooled_output = self._safe_l2_normalize(embeddings, dim=-1)

        # === TOKEN-LEVEL EMBEDDINGS (mới) ===
        token_outputs = None
        if return_tokens in ["tokens", "both"] and self.enable_late_interaction:
            token_embeddings = self.token_projection(hidden_states)
            token_embeddings = self._safe_l2_normalize(token_embeddings, dim=-1)

            if self.late_interaction_mode == "modality_wise":
                visual_tokens = self.visual_token_proj(token_embeddings * visual_mask.unsqueeze(-1))
                text_tokens = self.text_token_proj(token_embeddings * text_mask.unsqueeze(-1))
                token_outputs = {
                    "visual_tokens": visual_tokens,
                    "text_tokens": text_tokens,
                    "visual_mask": visual_mask,
                    "text_mask": text_mask,
                    "attention_mask": attention_mask
                }
            else:
                token_outputs = {
                    "tokens": token_embeddings,
                    "attention_mask": attention_mask,
                    "visual_mask": visual_mask,
                    "text_mask": text_mask
                }

        # === XÓA TENSOR ĐỂ GIẢM VRAM ===
        del inputs, outputs, hidden_states, visual_mask, text_mask
        if self.debug or _check_vram_threshold():
            torch.cuda.empty_cache()

        # === TRẢ VỀ THEO YÊU CẦU ===
        if return_tokens == "pooled":
            return pooled_output
        elif return_tokens == "tokens":
            return token_outputs
        else:  # "both"
            return {
                "pooled": pooled_output,
                "tokens": token_outputs
            }

    # === CẬP NHẬT: encode_query, encode_item ===
    def encode_query(self, images: List[Image.Image], texts: List[str], mode: str = "both", return_tokens: Optional[str] = None) -> Union[torch.Tensor, Dict]:
        return self._encode_batch(images, texts, mode=mode, is_training=False, return_tokens=return_tokens)

    def encode_item(self, images: List[Image.Image], texts: List[str], mode: str = "both", return_tokens: Optional[str] = None) -> Union[torch.Tensor, Dict]:
        return self._encode_batch(images, texts, mode=mode, is_training=False, return_tokens=return_tokens)

    # === CẬP NHẬT: forward ===
    def forward(self, query_images: List[Image.Image], query_texts: List[str],
                positive_images: List[Image.Image], positive_texts: List[str],
                negative_images: List[Image.Image], negative_texts: List[str],
                return_tokens: Optional[str] = None) -> Dict[str, Any]:
        if return_tokens is None:
            return_tokens = "both" if self.enable_late_interaction else "pooled"

        query_emb = self.encode_query(query_images, query_texts, return_tokens=return_tokens)
        positive_emb = self.encode_item(positive_images, positive_texts, return_tokens=return_tokens)
        negative_emb = self.encode_item(negative_images, negative_texts, return_tokens=return_tokens)

        result = {
            "query": query_emb,
            "positive": positive_emb,
            "negative": negative_emb
        }
        if self.enable_late_interaction:
            result["late_interaction_ready"] = True
        return result

    # === THÊM: Late Interaction Scoring ===
    def compute_late_interaction_score(self, query_tokens, doc_tokens,
                                     query_mask=None, doc_mask=None, mode="context"):
        if mode == "context":
            return self._contextualized_late_interaction(query_tokens, doc_tokens, query_mask, doc_mask)
        elif mode == "modality_wise":
            return self._modality_wise_late_interaction(query_tokens, doc_tokens, query_mask, doc_mask)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _contextualized_late_interaction(self, query_tokens, doc_tokens, query_mask, doc_mask):
        batch_size, n_query, dim = query_tokens.shape
        _, n_doc, _ = doc_tokens.shape
        scores = torch.zeros(batch_size, device=query_tokens.device)

        for b in range(batch_size):
            q_mask = query_mask[b] if query_mask is not None else torch.ones(n_query, device=query_tokens.device)
            d_mask = doc_mask[b] if doc_mask is not None else torch.ones(n_doc, device=doc_tokens.device)

            valid_q = query_tokens[b][q_mask.bool()]
            valid_d = doc_tokens[b][d_mask.bool()]

            if valid_q.numel() == 0 or valid_d.numel() == 0:
                continue

            sim_matrix = torch.mm(valid_q, valid_d.T)
            max_sims = sim_matrix.max(dim=1)[0]
            scores[b] = max_sims.sum()

        return scores / self.late_interaction_temperature

    def _modality_wise_late_interaction(self, query_tokens, doc_tokens, query_mask, doc_mask):
        visual_scores = self._contextualized_late_interaction(
            query_tokens["visual_tokens"], doc_tokens["visual_tokens"],
            query_mask, doc_tokens["visual_mask"]
        )
        text_scores = self._contextualized_late_interaction(
            query_tokens["text_tokens"], doc_tokens["text_tokens"],
            query_mask, doc_tokens["text_mask"]
        )
        return torch.max(visual_scores, text_scores)

    # === CẬP NHẬT: compute_triplet_loss ===
    def compute_triplet_loss(self, embeddings: Dict[str, Any],
                           margin: float = 0.2, temperature: float = 0.07,
                           nce_weight: float = 0.5, use_late_interaction: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        if use_late_interaction is None:
            use_late_interaction = self.enable_late_interaction

        if not use_late_interaction or not self.enable_late_interaction:
            # === GIỮ NGUYÊN: POOLED LOSS ===
            query_emb = embeddings["query"] if isinstance(embeddings["query"], torch.Tensor) else embeddings["query"]["pooled"]
            positive_emb = embeddings["positive"] if isinstance(embeddings["positive"], torch.Tensor) else embeddings["positive"]["pooled"]
            negative_emb = embeddings["negative"] if isinstance(embeddings["negative"], torch.Tensor) else embeddings["negative"]["pooled"]
            triplet_loss = triplet_cosine_loss(query_emb, positive_emb, negative_emb, margin)
            nce_loss = nce_inbatch_loss(query_emb, positive_emb, temperature)
            total_loss = triplet_loss + nce_weight * nce_loss
            return {"total_loss": total_loss, "triplet_loss": triplet_loss, "nce_loss": nce_loss}
        else:
            # === MỚI: LATE INTERACTION LOSS ===
            q_tokens = embeddings["query"]["tokens"]
            p_tokens = embeddings["positive"]["tokens"]
            n_tokens = embeddings["negative"]["tokens"]

            pos_scores = self.compute_late_interaction_score(
                q_tokens, p_tokens, mode=self.late_interaction_mode
            )
            neg_scores = self.compute_late_interaction_score(
                q_tokens, n_tokens, mode=self.late_interaction_mode
            )

            triplet_loss = F.relu(margin + (1 - pos_scores) - (1 - neg_scores)).mean()
            logits = torch.cat([pos_scores.unsqueeze(1), neg_scores.unsqueeze(1)], dim=1)
            targets = torch.zeros(pos_scores.shape[0], device=logits.device, dtype=torch.long)
            nce_loss = F.cross_entropy(logits / temperature, targets)
            total_loss = triplet_loss + nce_weight * nce_loss

            return {
                "total_loss": total_loss,
                "triplet_loss": triplet_loss,
                "nce_loss": nce_loss,
                "late_interaction_scores": {
                    "positive": pos_scores.mean().item(),
                    "negative": neg_scores.mean().item()
                }
            }

    def _normalize_sample(self, sample: Dict[str, Any], device: torch.device, sample_type: str, cache_dir: str) -> Dict[str, Any]:
        if sample_type not in ["query", "doc"]:
            raise ValueError(f"Invalid sample_type: {sample_type}")
        result = {"img": None, "text": "", "id": str(sample.get("id", sample.get("product_id", "unknown")))}
        sid = result["id"]
        path_key = "Ir_path" if sample_type == "query" else "It_path"
        path = sample.get(path_key)
        if isinstance(path, str) and path and path != "None" and os.path.exists(path):
            try:
                result["img"] = self._to_chw_float01(path, sample_id=sid)
                result[path_key] = path
            except Exception as e:
                logger.warning(f"[norm:{sample_type}]({sid}) failed to load image from path {path}: {e}")
                result["img"] = None
        else:
            img = sample.get("img")
            if isinstance(img, (torch.Tensor, np.ndarray, Image.Image)):
                try:
                    result["img"] = self._to_chw_float01(img, sample_id=sid)
                    result[path_key] = sample.get(path_key, "<from_tensor>")
                except Exception as e:
                    logger.warning(f"[norm:{sample_type}]({sid}) failed to process preloaded image: {e}")
                    result["img"] = None
            else:
                logger.error(f"[norm:{sample_type}]({sid}) invalid {path_key}: {path}")
        if result["img"] is None:
            raise ValueError(f"[norm:{sample_type}]({sid}) no valid image")
        result["text"] = str(sample.get("text") or sample.get("title") or sample.get("query") or sample.get("caption") or " ")
        return result

    def save_model(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "projection_head": self.projection_head.state_dict(),
            "fusion_alpha": float(self.fusion_alpha.item()),
            "out_dim": self.out_dim,
            "hidden_size": self.hidden_size,
        }, os.path.join(save_dir, "qwen_triplet_encoder.pth"))
        if hasattr(self.vlm, "peft_config"):
            try:
                self.vlm.save_pretrained(os.path.join(save_dir, "vlm_lora"))
            except Exception as e:
                logger.warning(f"Save LoRA failed: {e}")

    def load_model(self, save_dir: str):
        ckpt_path = os.path.join(save_dir, "qwen_triplet_encoder.pth")
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found at {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.projection_head.load_state_dict(ckpt["projection_head"])
        self.fusion_alpha.data.fill_(ckpt["fusion_alpha"])
        lora_dir = os.path.join(save_dir, "vlm_lora")
        if os.path.exists(lora_dir):
            try:
                from peft import PeftModel
                self.vlm = PeftModel.from_pretrained(self.vlm, lora_dir).to(self.device)
            except Exception as e:
                logger.warning(f"Load LoRA failed: {e}")

    def _to_chw_float01(self, img_any: Any, sample_id: str = "unknown") -> torch.Tensor:
        try:
            if isinstance(img_any, Image.Image):
                img = img_any.convert("RGB")
            elif isinstance(img_any, str):
                if not os.path.exists(img_any):
                    raise ValueError(f"Image path not found: {img_any}")
                img = Image.open(img_any).convert("RGB")
            elif isinstance(img_any, torch.Tensor):
                t = img_any
                if t.ndim == 4: t = t.squeeze(0)
                if t.ndim != 3:
                    raise ValueError(f"Tensor must be 3D, got {t.shape}")
                if t.shape[-1] in (1, 3) and t.shape[0] not in (1, 3):
                    t = t.permute(2, 0, 1)
                if t.shape[0] == 1: t = t.repeat(3, 1, 1)
                if t.shape[0] != 3:
                    raise ValueError(f"Expected 1/3 channels, got {t.shape[0]}")
                if t.dtype != torch.float32: t = t.to(torch.float32)
                if t.max() > 1.5: t = t / 255.0
                t = t.clamp(0, 1)
                return t
            elif isinstance(img_any, np.ndarray):
                if img_any.max() <= 1: img_any = (img_any * 255).astype(np.uint8)
                img = Image.fromarray(img_any).convert("RGB")
            else:
                raise ValueError(f"Unsupported type: {type(img_any)}")

            arr = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

        except Exception as e:
            logger.error(f"[_to_chw_float01]({sample_id}) failed: {e}")
            raise

    def encode_single(self, image: Any, text: str) -> torch.Tensor:
        pil_img = self._to_chw_float01(image).permute(1, 2, 0).cpu().numpy()
        pil_img = Image.fromarray((pil_img * 255).astype(np.uint8)).convert("RGB")
        return self._encode_batch([pil_img], [text or " "], mode="both", is_training=False).squeeze(0)

    def encode_image_only(self, image: Any) -> torch.Tensor:
        pil_img = self._to_chw_float01(image).permute(1, 2, 0).cpu().numpy()
        pil_img = Image.fromarray((pil_img * 255).astype(np.uint8)).convert("RGB")
        return self._encode_batch([pil_img], [""], mode="image", is_training=False).squeeze(0)

    def encode_text_only(self, text: str) -> torch.Tensor:
        dummy_pil = Image.new("RGB", (1, 1))
        return self._encode_batch([dummy_pil], [text or " "], mode="text", is_training=False).squeeze(0)
    
    def encode(self, image_path_or_obj, text, mode="both", return_tokens=None):
        """
        Backward compatibility method for evaluation.py
        Single image/text encoding wrapper
        """
        if isinstance(image_path_or_obj, str):
            # Load image from path
            pil_img = self._to_chw_float01(image_path_or_obj).permute(1, 2, 0).cpu().numpy()
            pil_img = Image.fromarray((pil_img * 255).astype(np.uint8)).convert("RGB")
        else:
            pil_img = image_path_or_obj

        if return_tokens is None:
            return_tokens = "both" if self.enable_late_interaction else "pooled"

        result = self._encode_batch([pil_img], [text or " "], mode=mode,
                                is_training=False, return_tokens=return_tokens)

        if isinstance(result, dict):
            return result
        else:
            return result.squeeze(0)

MultimodalRetriever = QwenVLTripletEncoder