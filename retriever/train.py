import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    try:
        xc = x.detach()
        return {
            "shape0": int(xc.shape[0]) if xc.ndim > 0 else 0,
            "min": float(torch.nan_to_num(xc).min().item()),
            "max": float(torch.nan_to_num(xc).max().item()),
            "mean": float(torch.nan_to_num(xc).float().mean().item()),
            "std": float(torch.nan_to_num(xc).float().std().item()),
            "nan": float(torch.isnan(xc).sum().item()),
            "inf": float(torch.isinf(xc).sum().item()),
            "norm_mean": float(torch.linalg.vector_norm(xc.reshape(xc.shape[0], -1), dim=1).mean().item()) if xc.ndim >= 2 else float(torch.linalg.vector_norm(xc).item()),
        }
    except Exception:
        return {"shape0": -1, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "nan": 0.0, "inf": 0.0, "norm_mean": 0.0}

def _write_jsonl(path: str, payload: Union[Dict[str, Any], List[Dict[str, Any]]]):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payloads = [payload] if isinstance(payload, dict) else list(payload)
        with open(path, "a", encoding="utf-8") as f:
            for p in payloads:
                p["cwd"] = os.getcwd()
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[debug] cannot write jsonl to {path}: {e}")

def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === HỖ TRỢ: Extract pooled embedding ===
def _extract_pooled(emb: Union[torch.Tensor, Dict]) -> torch.Tensor:
    return emb["pooled"] if isinstance(emb, dict) else emb

# === CẬP NHẬT: _encode_and_loss (HỖ TRỢ LATE INTERACTION) ===
def _encode_and_loss(
    model, q_imgs, q_txts, p_imgs, p_txts, n_imgs, n_txts,
    mode, margin, temp, nce_w,
    use_late_interaction: bool = False
):
    return_tokens = "both" if use_late_interaction else "pooled"

    q = model._encode_batch(q_imgs, q_txts, mode=mode, return_tokens=return_tokens)
    p = model._encode_batch(p_imgs, p_txts, mode=mode, return_tokens=return_tokens)
    n = model._encode_batch(n_imgs, n_txts, mode=mode, return_tokens=return_tokens)

    embeddings = {"query": q, "positive": p, "negative": n}
    L = model.compute_triplet_loss(
        embeddings,
        margin=margin, temperature=temp, nce_weight=nce_w,
        use_late_interaction=use_late_interaction
    )

    # Cosine similarity chỉ dùng pooled
    q_pooled = _extract_pooled(q)
    p_pooled = _extract_pooled(p)
    n_pooled = _extract_pooled(n)

    cos_pos = torch.nn.functional.cosine_similarity(q_pooled, p_pooled, dim=1).mean().item()
    cos_neg = torch.nn.functional.cosine_similarity(q_pooled, n_pooled, dim=1).mean().item()

    # Ghi thêm late interaction score
    if use_late_interaction and "late_interaction_scores" in L:
        L["late_pos_score"] = L["late_interaction_scores"]["positive"]
        L["late_neg_score"] = L["late_interaction_scores"]["negative"]

    return L, cos_pos, cos_neg

# === CẬP NHẬT: validate_model (HỖ TRỢ LATE INTERACTION) ===
def validate_model(
    model: nn.Module, val_loader: DataLoader, device: torch.device, triplet_margin: float,
    nce_temperature: float, nce_weight: float, debug: bool, debug_dir: str, debug_batches: int,
    use_autocast: bool, lambda_both: float = 1.0, lambda_text: float = 0.7, lambda_image: float = 0.7,
    enable_late_interaction_val: bool = False,
    current_epoch: int = 0,
    late_interaction_warmup_epochs: int = 1
) -> Dict[str, float]:
    assert (lambda_both + lambda_text + lambda_image) > 0, "All lambda_* are zero."
    model.eval()
    total_loss = total_triplet_loss = total_nce_loss = 0.0
    total_cos_pos = total_cos_neg = 0.0
    n_batches = 0
    debug_jsonl = os.path.join(debug_dir, "triplet_val_debug.jsonl")

    use_late = (
        enable_late_interaction_val and
        hasattr(model, 'enable_late_interaction') and
        model.enable_late_interaction and
        current_epoch >= late_interaction_warmup_epochs
    )

    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            try:
                q_imgs, q_txts = batch["query_images"], batch["query_texts"]
                p_imgs, p_txts = batch["positive_images"], batch["positive_texts"]
                n_imgs, n_txts = batch["negative_images"], batch["negative_texts"]
                modes = batch.get("modes", ["both"] * len(q_imgs))
                idx_both = [i for i, m in enumerate(modes) if m == "both"]
                idx_text = [i for i, m in enumerate(modes) if m == "text"]
                idx_image = [i for i, m in enumerate(modes) if m == "image"]
                def take(xs, idx): return [xs[i] for i in idx]

                total_loss_batch = None
                losses_detail = {}
                cos_pos_dict = {}
                cos_neg_dict = {}

                with torch.cuda.amp.autocast(enabled=use_autocast and torch.cuda.is_available()):
                    for mode_name, idx_list, lambda_w in [
                        ("both", idx_both, lambda_both),
                        ("text", idx_text, lambda_text),
                        ("image", idx_image, lambda_image)
                    ]:
                        if not idx_list:
                            continue

                        L, cos_pos, cos_neg = _encode_and_loss(
                            model,
                            take(q_imgs, idx_list), take(q_txts, idx_list),
                            take(p_imgs, idx_list), take(p_txts, idx_list),
                            take(n_imgs, idx_list), take(n_txts, idx_list),
                            mode=mode_name,
                            margin=triplet_margin, temp=nce_temperature, nce_w=nce_weight,
                            use_late_interaction=use_late
                        )

                        if torch.isnan(L["total_loss"]) or torch.isinf(L["total_loss"]):
                            logger.warning(f"[val s{step}] NaN/Inf in '{mode_name}' mode -> skip")
                            continue

                        loss_contrib = L["total_loss"] * lambda_w
                        total_loss_batch = loss_contrib if total_loss_batch is None else total_loss_batch + loss_contrib
                        losses_detail[mode_name] = L
                        cos_pos_dict[mode_name] = cos_pos
                        cos_neg_dict[mode_name] = cos_neg

                if total_loss_batch is None:
                    continue

                total_loss += float(total_loss_batch.item())
                for m, w in [("both", lambda_both), ("text", lambda_text), ("image", lambda_image)]:
                    if m in losses_detail:
                        total_triplet_loss += float(losses_detail[m].get("triplet_loss", 0.0)) * w
                        total_nce_loss += float(losses_detail[m].get("nce_loss", 0.0)) * w

                valid_cos_pos = [v for k, v in cos_pos_dict.items()]
                valid_cos_neg = [v for k, v in cos_neg_dict.items()]
                total_cos_pos += sum(valid_cos_pos) / max(1, len(valid_cos_pos))
                total_cos_neg += sum(valid_cos_neg) / max(1, len(valid_cos_neg))

                n_batches += 1

                if debug and step < debug_batches:
                    log_entry = {
                        "kind": "val_step", "step": step, "epoch": current_epoch + 1,
                        "total_loss": float(total_loss_batch.item()),
                        "use_late_interaction": use_late
                    }
                    for m in ["both", "text", "image"]:
                        if m in losses_detail:
                            log_entry.update({
                                f"loss_{m}": float(losses_detail[m]["total_loss"]),
                                f"cos_pos_{m}": float(cos_pos_dict.get(m, 0.0)),
                                f"cos_neg_{m}": float(cos_neg_dict.get(m, 0.0)),
                            })
                            if "late_pos_score" in losses_detail[m]:
                                log_entry[f"late_pos_{m}"] = losses_detail[m]["late_pos_score"]
                    _write_jsonl(debug_jsonl, log_entry)

            except Exception as e:
                query_base_ids = batch.get("query_base_ids", ["unknown"])[:2]
                logger.error(f"[val s{step}] Error: {e}")
                _write_jsonl(debug_jsonl, {"kind": "val_error", "step": step, "error": str(e)})
                continue

    if n_batches == 0:
        return {"val_loss": float('inf'), "val_triplet_loss": float('inf'), "val_nce_loss": float('inf'), "cos_pos": 0.0, "cos_neg": 0.0}

    return {
        "val_loss": total_loss / n_batches,
        "val_triplet_loss": total_triplet_loss / n_batches,
        "val_nce_loss": total_nce_loss / n_batches,
        "cos_pos": total_cos_pos / n_batches,
        "cos_neg": total_cos_neg / n_batches
    }

# === CẬP NHẬT: train_qwen_triplet_retriever (HOÀN CHỈNH) ===
def train_qwen_triplet_retriever(
    model, train_loader: DataLoader, val_loader: DataLoader, documents: List[Dict[str, Any]], *,
    epochs: int = 5, lr: float = 1e-4, weight_decay: float = 1e-5, warmup_steps: int = 500,
    device: str = "cuda", save_dir: str = "checkpoints", no_improve_epochs: int = 2, debug: bool = False,
    debug_batches: int = 2, debug_dir: str = "cache/debug", skip_on_error: bool = True,
    triplet_margin: float = 0.2, nce_temperature: float = 0.07, nce_weight: float = 0.5,
    grad_clip_norm: float = 1.0, use_autocast: bool = True, hard_neg_start_epoch: int = 2,
    lambda_both: float = 1.0, lambda_text: float = 0.7, lambda_image: float = 0.7,
    min_delta: float = 1e-3, hard_neg_index: Optional[Dict[str, List[str]]] = None,
    # === THÊM MỚI: LATE INTERACTION CONTROL ===
    enable_late_interaction_training: bool = False,
    late_interaction_warmup_epochs: int = 1
) -> Dict[str, Any]:
    assert (lambda_both + lambda_text + lambda_image) > 0, "All lambda_* are zero."
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    debug_jsonl = os.path.join(debug_dir, "triplet_train_debug.jsonl")
    if os.path.exists(debug_jsonl):
        try:
            os.remove(debug_jsonl)
            logger.info(f"[train] Cleared old debug log: {debug_jsonl}")
        except Exception:
            pass

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.train()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # === KÍCH HOẠT LATE INTERACTION TRAINING MODE ===
    if enable_late_interaction_training and hasattr(model, 'enable_late_interaction'):
        model.enable_late_interaction = True
        logger.info("Enabled late interaction training mode")
        if hasattr(model, 'late_interaction_temperature'):
            logger.info(f"Late interaction temperature: {model.late_interaction_temperature.item():.4f}")
        logger.info(f"Late interaction mode: {model.late_interaction_mode}")
    else:
        model.enable_late_interaction = False
        logger.info("Late interaction training mode is DISABLED")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found")
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)) if warmup_steps > 0 else 1.0
    )
    scaler = torch.amp.GradScaler('cuda') if use_autocast and torch.cuda.is_available() else None

    if debug:
        torch.autograd.set_detect_anomaly(True)
        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = model.count_trainable_params()
        logger.info(f"[debug] Parameters: total={n_total:,} trainable={n_trainable:,} device={dev.type}")

    best_loss = float('inf')
    best_checkpoint = None
    no_improve_count = 0
    global_step = 0
    skipped_batches = 0
    hard_neg_index = {} if hard_neg_index is None else hard_neg_index

    for epoch in range(epochs):
        model.train()
        total_loss = total_triplet_loss = total_nce_loss = 0.0
        n_batches = 0

        # === TWO-STAGE TRAINING STRATEGY ===
        if epoch < late_interaction_warmup_epochs:
            model.enable_late_interaction = False
            logger.info(f"[Epoch {epoch+1}/{epochs}] WARM-UP stage: Using pooled embeddings")
            use_late_interaction = False
        else:
            if enable_late_interaction_training and hasattr(model, 'enable_late_interaction'):
                model.enable_late_interaction = True
                logger.info(f"[Epoch {epoch+1}/{epochs}] FINE-TUNE stage: Using late interaction")
                use_late_interaction = True
            else:
                model.enable_late_interaction = False
                use_late_interaction = False

        # === HARD NEGATIVE MINING ===
        if epoch >= hard_neg_start_epoch and not hard_neg_index:
            logger.info(f"[epoch {epoch+1}] Computing hard negative index")
            try:
                from faiss import IndexFlatIP
                model.eval()
                embeddings = []
                pids = []
                for doc in documents:
                    emb = model._encode_batch([doc["It_path"]], [doc["text"]], mode="both", return_tokens="pooled").detach().cpu().numpy()
                    embeddings.append(emb[0])
                    pids.append(doc["id"])
                index = IndexFlatIP(embeddings[0].shape[0])
                index.add(np.array(embeddings))
                _, indices = index.search(np.array(embeddings), 5)
                hard_neg_index.clear()
                hard_neg_index.update({pid: [pids[j] for j in idx[1:]] for pid, idx in zip(pids, indices)})
                model.train()
            except Exception as e:
                logger.warning(f"[epoch {epoch+1}] Failed to compute hard negatives: {e}")
                hard_neg_index.clear()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for step, batch in enumerate(pbar):
            try:
                q_imgs, q_txts = batch["query_images"], batch["query_texts"]
                p_imgs, p_txts = batch["positive_images"], batch["positive_texts"]
                n_imgs, n_txts = batch["negative_images"], batch["negative_texts"]
                modes = batch.get("modes", ["both"] * len(q_imgs))
                idx_both = [i for i, m in enumerate(modes) if m == "both"]
                idx_text = [i for i, m in enumerate(modes) if m == "text"]
                idx_image = [i for i, m in enumerate(modes) if m == "image"]
                def take(xs, idx): return [xs[i] for i in idx]

                total_loss_batch = None
                losses_detail = {}
                cos_pos_dict = {}
                cos_neg_dict = {}

                with torch.cuda.amp.autocast(enabled=use_autocast and torch.cuda.is_available()):
                    for mode_name, idx_list, lambda_w in [
                        ("both", idx_both, lambda_both),
                        ("text", idx_text, lambda_text),
                        ("image", idx_image, lambda_image)
                    ]:
                        if not idx_list:
                            continue

                        L, cos_pos, cos_neg = _encode_and_loss(
                            model,
                            take(q_imgs, idx_list), take(q_txts, idx_list),
                            take(p_imgs, idx_list), take(p_txts, idx_list),
                            take(n_imgs, idx_list), take(n_txts, idx_list),
                            mode=mode_name,
                            margin=triplet_margin, temp=nce_temperature, nce_w=nce_weight,
                            use_late_interaction=use_late_interaction
                        )

                        if torch.isnan(L["total_loss"]) or torch.isinf(L["total_loss"]):
                            logger.warning(f"[epoch{epoch+1}][step{step}] NaN/Inf in '{mode_name}' mode -> skip")
                            continue

                        loss_contrib = L["total_loss"] * lambda_w
                        total_loss_batch = loss_contrib if total_loss_batch is None else total_loss_batch + loss_contrib
                        losses_detail[mode_name] = L
                        cos_pos_dict[mode_name] = cos_pos
                        cos_neg_dict[mode_name] = cos_neg

                if total_loss_batch is None:
                    if skip_on_error:
                        skipped_batches += 1
                        continue
                    else:
                        raise ValueError("Empty batch after mode split")

                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(total_loss_batch).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss_batch.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
                    optimizer.step()
                scheduler.step()

                total_loss += float(total_loss_batch.item())
                for m, w in [("both", lambda_both), ("text", lambda_text), ("image", lambda_image)]:
                    if m in losses_detail:
                        total_triplet_loss += float(losses_detail[m].get("triplet_loss", 0.0)) * w
                        total_nce_loss += float(losses_detail[m].get("nce_loss", 0.0)) * w
                n_batches += 1
                global_step += 1

                if debug and step < debug_batches:
                    log_entry = {
                        "kind": "train_step", "epoch": epoch + 1, "step": step,
                        "total_loss": float(total_loss_batch.item()),
                        "use_late_interaction": use_late_interaction,
                        "fusion_alpha": float(model.fusion_alpha.item()), "lr": float(scheduler.get_last_lr()[0])
                    }
                    for m in ["both", "text", "image"]:
                        if m in losses_detail:
                            log_entry.update({
                                f"loss_{m}": float(losses_detail[m]["total_loss"]),
                                f"cos_pos_{m}": float(cos_pos_dict.get(m, 0.0)),
                                f"cos_neg_{m}": float(cos_neg_dict.get(m, 0.0)),
                            })
                    _write_jsonl(debug_jsonl, log_entry)

                pbar.set_postfix({
                    'loss': f'{total_loss_batch.item():.4f}',
                    'alpha': f'{model.fusion_alpha.item():.3f}'
                })

            except Exception as e:
                query_base_ids = batch.get("query_base_ids", ["unknown"])[:2]
                logger.error(f"[epoch{epoch+1}][step{step}] Error: {e}")
                _write_jsonl(debug_jsonl, {"kind": "train_error", "epoch": epoch + 1, "step": step, "error": str(e)})
                if skip_on_error:
                    skipped_batches += 1
                    continue
                else:
                    raise

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(1, n_batches)
        logger.info(f"[epoch {epoch+1}/{epochs}] avg_loss={avg_loss:.6f} processed_batches={n_batches} skipped={skipped_batches}")

        # === VALIDATION ===
        val_metrics = validate_model(
            model=model, val_loader=val_loader, device=dev,
            triplet_margin=triplet_margin, nce_temperature=nce_temperature, nce_weight=nce_weight,
            debug=debug, debug_dir=debug_dir, debug_batches=debug_batches, use_autocast=use_autocast,
            lambda_both=lambda_both, lambda_text=lambda_text, lambda_image=lambda_image,
            enable_late_interaction_val=enable_late_interaction_training,
            current_epoch=epoch,
            late_interaction_warmup_epochs=late_interaction_warmup_epochs
        )
        val_loss = val_metrics["val_loss"]
        logger.info(f"[epoch {epoch+1}] validation loss: {val_loss:.6f}")

        # === EARLY STOPPING ===
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            no_improve_count = 0
            ep_dir = os.path.join(save_dir, f"qwen_triplet_epoch_{epoch+1}")
            os.makedirs(ep_dir, exist_ok=True)
            model.save_model(ep_dir)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(ep_dir, 'checkpoint.pt'))
            best_checkpoint = ep_dir
            logger.info(f"[epoch {epoch+1}] New best loss={best_loss:.6f} -> saved {ep_dir}")
        else:
            no_improve_count += 1
            if no_improve_count >= no_improve_epochs:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    return {
        "best_loss": best_loss,
        "best_checkpoint": best_checkpoint,
        "final_fusion_alpha": float(model.fusion_alpha.item()),
        "final_val_metrics": val_metrics
    }