import os
import json
import random
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict, Counter
from functools import lru_cache
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageFile

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset for Qwen2.5-VL Triplet Learning
class QwenVLTripletDataset(Dataset):
    """Dataset for Qwen2.5-VL triplet learning."""
    def __init__(self, data: List[Tuple[Dict, Dict, List[Dict]]], cache_dir: str = "cache"):
        self.data = data
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

@lru_cache(maxsize=64)
def _load_image_cached(image_path: str) -> Image.Image:
    """Load and cache PIL image from path."""
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"[load_image_cached] Failed to load {image_path}: {e}")
        raise

# === INFER CATEGORY FROM TEXT ===
def infer_category(text: str) -> str:
    """Infer category from title/caption using keyword matching."""
    text = text.lower().strip()
    if not text:
        return "other"

    keyword_map = {
        "shoe": ["shoe", "sneaker", "boot", "sandal", "heel", "loafer"],
        "dress": ["dress", "gown", "frock", "skirt", "maxi"],
        "bag": ["bag", "handbag", "backpack", "purse", "tote", "clutch"],
        "shirt": ["shirt", "t-shirt", "blouse", "top", "tee", "polo"],
        "pant": ["pant", "jean", "trouser", "short", "legging", "jogger"],
        "jacket": ["jacket", "coat", "blazer", "hoodie", "sweater"],
        "electronics": ["phone", "laptop", "camera", "headphone", "earphone", "tv", "speaker"],
        "watch": ["watch", "smartwatch"],
    }

    scores = Counter()
    for cat, keywords in keyword_map.items():
        scores[cat] = sum(1 for kw in keywords if kw in text)

    if scores:
        return scores.most_common(1)[0][0]
    return "other"

# === PREPROCESS TEXT ===
def preprocess_text(text: str) -> str:
    text = (text or " ").strip()[:256]
    text = " ".join(word for word in text.split() if not (word.startswith("sku") and word[3:].isdigit()))
    return text

# === WRITE JSONL (SUPPORT DICT & LIST) ===
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
        logger.warning(f"Cannot write jsonl to {path}: {e}")

# === BASIC COLLATE ===
def triplet_collate_basic(batch, cache_dir: str, debug: bool = False):
    valid_samples = []
    skip_reasons = defaultdict(int)

    for i, (q, p, n_list) in enumerate(batch):
        sid = str(q.get("id", q.get("product_id", "unknown")))
        try:
            q_path = q.get("Ir_path")
            if not isinstance(q_path, str) or not os.path.exists(q_path):
                skip_reasons["query_image_missing"] += 1
                if debug:
                    logger.warning(f"[basic_collate] Skip triplet {i}: invalid query image path {q_path}, q_id={sid}")
                    _write_jsonl(os.path.join(cache_dir, "collate_skips.jsonl"), [{
                        "type": "basic_collate_skip", "triplet_idx": i, "q_id": sid, "reason": "query_image_missing", "q_path": q_path
                    }])
                continue
            try:
                q_img = _load_image_cached(q_path)
            except Exception as e:
                skip_reasons["query_image_load_error"] += 1
                if debug:
                    logger.warning(f"[basic_collate] Skip triplet {i}: failed to load query image {q_path}, q_id={sid}, error={e}")
                    _write_jsonl(os.path.join(cache_dir, "collate_skips.jsonl"), [{
                        "type": "basic_collate_skip", "triplet_idx": i, "q_id": sid, "reason": "query_image_load_error", "q_path": q_path, "error": str(e)
                    }])
                continue
            q_text = preprocess_text(str(q.get("caption") or " "))
            q_id = str(q.get("id", q.get("product_id", "unknown")))
            q_category = str(q.get("category", "unknown"))  # category_1

            p_path = p.get("It_path", p.get("Ir_path"))
            if not isinstance(p_path, str) or not os.path.exists(p_path):
                skip_reasons["pos_image_missing"] += 1
                if debug:
                    logger.warning(f"[basic_collate] Skip triplet {i}: invalid positive image path {p_path}, q_id={sid}")
                    _write_jsonl(os.path.join(cache_dir, "collate_skips.jsonl"), [{
                        "type": "basic_collate_skip", "triplet_idx": i, "q_id": sid, "reason": "pos_image_missing", "p_path": p_path
                    }])
                continue
            try:
                p_img = _load_image_cached(p_path)
            except Exception as e:
                skip_reasons["pos_image_load_error"] += 1
                if debug:
                    logger.warning(f"[basic_collate] Skip triplet {i}: failed to load positive image {p_path}, q_id={sid}, error={e}")
                    _write_jsonl(os.path.join(cache_dir, "collate_skips.jsonl"), [{
                        "type": "basic_collate_skip", "triplet_idx": i, "q_id": sid, "reason": "pos_image_load_error", "p_path": p_path, "error": str(e)
                    }])
                continue
            p_text = preprocess_text(str(p.get("text") or p.get("title") or p.get("caption") or " "))
            p_id = str(p.get("id", p.get("product_id", "unknown")))
            p_category = str(p.get("category", q_category))  # category_2

            sample = {
                "q_img": q_img, "q_text": q_text, "q_id": q_id, "q_path": q_path, "q_category": q_category,
                "p_img": p_img, "p_text": p_text, "p_id": p_id, "p_path": p_path, "p_category": p_category
            }
            if not all(k in sample for k in ["q_img", "q_text", "q_id", "q_path", "q_category", "p_img", "p_text", "p_id", "p_path", "p_category"]):
                skip_reasons["invalid_sample_format"] += 1
                if debug:
                    logger.warning(f"[basic_collate] Skip triplet {i}: invalid sample format, q_id={sid}")
                    _write_jsonl(os.path.join(cache_dir, "collate_skips.jsonl"), [{
                        "type": "basic_collate_skip", "triplet_idx": i, "q_id": sid, "reason": "invalid_sample_format"
                    }])
                continue
            valid_samples.append(sample)
        except Exception as e:
            skip_reasons[str(e)] += 1
            if debug:
                logger.warning(f"[basic_collate] Skip triplet {i}: {e}, q_id={sid}")
                _write_jsonl(os.path.join(cache_dir, "collate_skips.jsonl"), [{
                    "type": "basic_collate_skip", "triplet_idx": i, "q_id": sid, "reason": str(e), "q_path": q.get("Ir_path", ""), "p_path": p.get("It_path", p.get("Ir_path", ""))
                }])
            continue

    if not valid_samples:
        logger.error(f"[basic_collate] Empty batch, skip reasons: {dict(skip_reasons)}")
        raise ValueError(f"No valid triplets in batch, skip reasons: {dict(skip_reasons)}")

    return valid_samples, skip_reasons

# === MULTIMODAL COLLATE ===
def triplet_collate_multimodal(batch, model, device, cache_dir, all_pids=None, pid2doc=None, hard_neg_index=None, debug=False, is_val_or_test=False, seed=42):
    queries, pos_docs, neg_docs = [], [], []
    modes, query_paths, pos_paths, neg_paths = [], [], [], []
    query_categories, pos_categories, neg_categories = [], [], []
    query_base_ids, pos_base_ids, neg_base_ids = [], [], []
    DUMMY_IMG = Image.new("RGB", (1, 1), (0, 0, 0))
    MODALITY_DROPOUT_PROB = 0.15
    P_B, P_C = (0.0, 0.0) if is_val_or_test else (0.25, 0.25)

    valid_samples, skip_reasons = triplet_collate_basic(batch, cache_dir, debug)
    valid_samples = [s for s in valid_samples if isinstance(s, dict) and all(k in s for k in ["q_id", "p_id", "q_category", "p_category"])]
    if not valid_samples:
        logger.error(f"[multimodal_collate] No valid samples after filtering, skip reasons: {dict(skip_reasons)}")
        raise ValueError("No valid samples in batch")

    if is_val_or_test and len(valid_samples) < len(batch) * 0.5:
        logger.warning(f"[multimodal_collate] High skip rate in val/test: kept {len(valid_samples)}/{len(batch)}, reasons: {dict(skip_reasons)}")
        _write_jsonl(os.path.join(cache_dir, "collate_stats.jsonl"), {
            "type": "batch_stats", "batch_size": len(valid_samples), "skip_reasons": dict(skip_reasons), "is_val_or_test": is_val_or_test
        })

    # Generate A, B, C samples
    for sample in valid_samples:
        created_modes_for_this_anchor = set()
        drop_mode = 0.0 if is_val_or_test else random.random()

        if drop_mode < MODALITY_DROPOUT_PROB / 2 and "text" not in created_modes_for_this_anchor:
            queries.append({"img": DUMMY_IMG, "text": sample["q_text"], "id": f"{sample['q_id']}_text", "base_id": sample["q_id"]})
            pos_docs.append({"img": DUMMY_IMG, "text": sample["p_text"], "id": f"{sample['p_id']}_text", "base_id": sample["p_id"]})
            modes.append("text")
            created_modes_for_this_anchor.add("text")
        elif drop_mode < MODALITY_DROPOUT_PROB and "image" not in created_modes_for_this_anchor:
            queries.append({"img": sample["q_img"], "text": " ", "id": f"{sample['q_id']}_image", "base_id": sample["q_id"]})
            pos_docs.append({"img": sample["p_img"], "text": " ", "id": f"{sample['p_id']}_image", "base_id": sample["p_id"]})
            modes.append("image")
            created_modes_for_this_anchor.add("image")
        else:
            queries.append({"img": sample["q_img"], "text": sample["q_text"], "id": sample["q_id"], "base_id": sample["q_id"]})
            pos_docs.append({"img": sample["p_img"], "text": sample["p_text"], "id": sample["p_id"], "base_id": sample["p_id"]})
            modes.append("both")
            created_modes_for_this_anchor.add("both")

        if random.random() < P_B and "text" not in created_modes_for_this_anchor:
            queries.append({"img": DUMMY_IMG, "text": sample["q_text"], "id": f"{sample['q_id']}_text", "base_id": sample["q_id"]})
            pos_docs.append({"img": DUMMY_IMG, "text": sample["p_text"], "id": f"{sample['p_id']}_text", "base_id": sample["p_id"]})
            modes.append("text")
            created_modes_for_this_anchor.add("text")

        if random.random() < P_C and "image" not in created_modes_for_this_anchor:
            queries.append({"img": sample["q_img"], "text": " ", "id": f"{sample['q_id']}_image", "base_id": sample["q_id"]})
            pos_docs.append({"img": sample["p_img"], "text": " ", "id": f"{sample['p_id']}_image", "base_id": sample["p_id"]})
            modes.append("image")
            created_modes_for_this_anchor.add("image")

        # Append metadata
        for _ in range(len(queries) - len(query_paths)):
            query_paths.append(sample["q_path"])
            pos_paths.append(sample["p_path"])
            query_categories.append(sample["q_category"])
            pos_categories.append(sample["p_category"])
            query_base_ids.append(sample["q_id"])
            pos_base_ids.append(sample["p_id"])

    # === NEGATIVE SAMPLING: DỰA VÀO category_2 (TARGET) ===
    for i in range(len(queries)):
        q_base_id = query_base_ids[i]
        p_base_id = pos_base_ids[i]
        target_category = pos_categories[i]  # ← DÙNG CATEGORY_2

        neg_img, neg_text, neg_id, neg_path, neg_category, neg_base_id = None, None, None, None, None, None

        # Hard negatives (same target category)
        if hard_neg_index and p_base_id in hard_neg_index and hard_neg_index[p_base_id]:
            cand_hard_neg_pids = [pid for pid in hard_neg_index[p_base_id] if pid != q_base_id and pid != p_base_id]
            cand_same_cat = [pid for pid in cand_hard_neg_pids if pid2doc.get(pid, {}).get("category", "unknown") == target_category]
            cand_neg_pids = cand_same_cat or cand_hard_neg_pids
            if cand_neg_pids:
                neg_pid = random.choice(cand_neg_pids)
                neg_doc = pid2doc.get(neg_pid)
                if neg_doc:
                    try:
                        neg_img = _load_image_cached(neg_doc["It_path"]) if modes[i] != "text" else DUMMY_IMG
                    except Exception:
                        skip_reasons["neg_image_load_error"] += 1
                        neg_img = pos_docs[i]["img"].copy()
                    neg_text = neg_doc["text"] if modes[i] != "image" else " "
                    neg_id = f"{neg_pid}_{modes[i]}"
                    neg_path = neg_doc["It_path"]
                    neg_category = neg_doc.get("category", target_category)
                    neg_base_id = neg_pid

        # In-batch negatives
        if neg_img is None:
            cand_neg = [
                j for j, s in enumerate(valid_samples)
                if s["p_id"] != q_base_id and s["p_id"] != p_base_id and s["p_category"] == target_category
            ]
            if not cand_neg:
                cand_neg = [j for j, s in enumerate(valid_samples) if s["p_id"] != q_base_id and s["p_id"] != p_base_id]
            if cand_neg:
                neg_idx = random.choice(cand_neg)
                neg_sample = valid_samples[neg_idx]
                neg_img = DUMMY_IMG if modes[i] == "text" else neg_sample["p_img"]
                neg_text = neg_sample["p_text"] if modes[i] != "image" else " "
                neg_id = f"{neg_sample['p_id']}_{modes[i]}"
                neg_path = neg_sample["p_path"]
                neg_category = neg_sample["p_category"]
                neg_base_id = neg_sample["p_id"]
            else:
                # Fallback
                if all_pids is None or not pid2doc:
                    neg_img = pos_docs[i]["img"].copy()
                    neg_text = pos_docs[i]["text"]
                    neg_id = f"{p_base_id}_dummy"
                    neg_path = pos_paths[i]
                    neg_category = pos_categories[i]
                    neg_base_id = p_base_id
                else:
                    cand_neg_pids = list(all_pids - {q_base_id, p_base_id})
                    cand_same_cat = [pid for pid in cand_neg_pids if pid2doc.get(pid, {}).get("category", "unknown") == target_category]
                    cand_neg_pids = cand_same_cat or cand_neg_pids
                    if not cand_neg_pids:
                        neg_img = pos_docs[i]["img"].copy()
                        neg_text = pos_docs[i]["text"]
                        neg_id = f"{p_base_id}_dummy"
                        neg_path = pos_paths[i]
                        neg_category = pos_categories[i]
                        neg_base_id = p_base_id
                    else:
                        neg_pid = random.choice(cand_neg_pids)
                        neg_doc = pid2doc.get(neg_pid)
                        if not neg_doc:
                            neg_img = pos_docs[i]["img"].copy()
                            neg_text = pos_docs[i]["text"]
                            neg_id = f"{p_base_id}_dummy"
                            neg_path = pos_paths[i]
                            neg_category = pos_categories[i]
                            neg_base_id = p_base_id
                        else:
                            try:
                                neg_img = _load_image_cached(neg_doc["It_path"]) if modes[i] != "text" else DUMMY_IMG
                            except Exception:
                                neg_img = pos_docs[i]["img"].copy()
                            neg_text = neg_doc["text"] if modes[i] != "image" else " "
                            neg_id = f"{neg_pid}_{modes[i]}"
                            neg_path = neg_doc["It_path"]
                            neg_category = neg_doc.get("category", target_category)
                            neg_base_id = neg_pid

        neg_docs.append({"img": neg_img, "text": neg_text, "id": neg_id, "base_id": neg_base_id})
        neg_paths.append(neg_path)
        neg_categories.append(neg_category)
        neg_base_ids.append(neg_base_id)

    if debug:
        _write_jsonl(os.path.join(cache_dir, "collate_stats.jsonl"), {
            "type": "batch_stats",
            "batch_size": len(queries),
            "a_count": modes.count('both'),
            "b_count": modes.count('text'),
            "c_count": modes.count('image'),
            "skip_reasons": {k: int(v) for k, v in skip_reasons.items()}
        })

    return {
        "query_images": [q["img"] for q in queries],
        "query_texts": [q["text"] for q in queries],
        "positive_images": [p["img"] for p in pos_docs],
        "positive_texts": [p["text"] for p in pos_docs],
        "negative_images": [n["img"] for n in neg_docs],
        "negative_texts": [n["text"] for n in neg_docs],
        "batch_size": len(queries),
        "query_ids": [q["id"] for q in queries],
        "positive_ids": [p["id"] for p in pos_docs],
        "negative_ids": [n["id"] for n in neg_docs],
        "query_base_ids": query_base_ids,
        "positive_base_ids": pos_base_ids,
        "negative_base_ids": neg_base_ids,
        "query_paths": query_paths,
        "positive_paths": pos_paths,
        "negative_paths": neg_paths,
        "query_categories": query_categories,
        "positive_categories": pos_categories,
        "negative_categories": neg_categories,
        "modes": modes
    }

# === WORKER INIT ===
def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)

# === PREPROCESS WCAPTIONS + INFER CATEGORY_1 & CATEGORY_2 ===
def preprocess_wcaptions(wcaptions: List[Dict], cache_dir: str = "cache", debug: bool = False) -> List[Dict]:
    processed = []
    required_fields = ["product_id_1", "Ir_path", "caption", "product_id_2", "It_path", "title_2"]

    for i, item in enumerate(wcaptions):
        if not isinstance(item, dict):
            if debug:
                _write_jsonl(os.path.join(cache_dir, "wcaptions_skips.jsonl"), {
                    "type": "wcaptions_invalid", "item_idx": i, "reason": "not a dict"
                })
            continue

        if not all(field in item for field in required_fields):
            if debug:
                _write_jsonl(os.path.join(cache_dir, "wcaptions_skips.jsonl"), {
                    "type": "wcaptions_invalid", "item_idx": i, "reason": "missing fields",
                    "missing": [k for k in required_fields if k not in item]
                })
            continue

        if not os.path.exists(item["Ir_path"]) or not os.path.exists(item["It_path"]):
            if debug:
                _write_jsonl(os.path.join(cache_dir, "wcaptions_skips.jsonl"), {
                    "type": "wcaptions_invalid", "item_idx": i, "reason": "invalid image paths",
                    "Ir_path": item.get("Ir_path"), "It_path": item.get("It_path")
                })
            continue

        # === INFER category_1 (ref) & category_2 (target) ===
        title_1 = str(item.get("title_1", "")).strip()
        caption = str(item.get("caption", "")).strip()
        title_2 = str(item.get("title_2", "")).strip()

        # category_1: ref
        cat1 = item.get("category_1") or item.get("category")
        item["category_1"] = str(cat1).strip() if cat1 and str(cat1).strip() != "unknown" else infer_category(title_1 + " " + caption)

        # category_2: target
        cat2 = item.get("category_2") or item.get("category")
        item["category_2"] = str(cat2).strip() if cat2 and str(cat2).strip() != "unknown" else infer_category(title_2 + " " + caption)

        processed.append(item)

    if debug:
        logger.info(f"[preprocess_wcaptions] Kept {len(processed)}/{len(wcaptions)} items")
        _write_jsonl(os.path.join(cache_dir, "wcaptions_stats.jsonl"), {
            "type": "wcaptions_stats", "total": len(wcaptions), "kept": len(processed)
        })
    return processed

# === SPLIT DATASET ===
def split_wcaptions_dataset(
    wcaptions: List[Dict],
    cache_dir: str = "cache",
    max_negatives: int = 5,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    batch_size: int = 128,
    num_workers: int = 2,
    seed: int = 42,
    device: str = "cuda",
    model=None,
    debug: bool = False,
    write_report: bool = True,
    report_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[Dict]]:
    if model is None:
        raise ValueError("model parameter (QwenVLTripletEncoder) is required")

    os.makedirs(cache_dir, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    if report_path is None:
        report_path = os.path.join(cache_dir, "triplet_debug_report.jsonl")

    if write_report and os.path.exists(report_path):
        try:
            os.remove(report_path)
        except Exception as e:
            logger.warning(f"[split] Cannot clear report {report_path}: {e}")

    def _report(payload: Dict[str, Any]):
        if not write_report: return
        try:
            payload["cwd"] = os.getcwd()
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[split] Cannot write report {report_path}: {e}")

    wcaptions = preprocess_wcaptions(wcaptions, cache_dir=cache_dir, debug=debug)

    documents_path = os.path.join(cache_dir, "triplet_documents.json")
    documents: List[Dict[str, Any]] = []
    pid2meta: Dict[str, Dict] = {}
    stats = {"ok": 0, "invalid_format": 0}

    for w in tqdm(wcaptions, desc="Building documents"):
        pid1 = str(w.get("product_id_1", ""))
        pid2 = str(w.get("product_id_2", pid1))
        it_path = w.get("It_path", w.get("Ir_path"))

        if not all(k in w for k in ["product_id_1", "Ir_path", "caption"]):
            stats["invalid_format"] += 1
            continue

        if pid2 not in pid2meta:
            text = str(w.get("title_2", w.get("title_1", ""))).strip() or str(w.get("caption", "")).strip() or " "
            pid2meta[pid2] = {
                "id": pid2,
                "It_path": it_path,
                "text": text,
                "category": w["category_2"]  # ← DÙNG category_2
            }
            stats["ok"] += 1

    documents = list(pid2meta.values())
    if len(documents) < 3:
        raise ValueError(f"Too few documents ({len(documents)})")

    try:
        with open(documents_path, "w", encoding="utf-8") as f:
            json.dump([{"id": d["id"], "It_path": d["It_path"], "text": d["text"], "category": d["category"]} for d in documents], f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[split] Cannot save documents.json: {e}")

    pid2doc = {d["id"]: d for d in documents}
    all_pids = set(pid2doc.keys())

    random.seed(seed)
    data: List[Tuple[Dict, Dict, List[Dict]]] = []
    skip_q = 0
    reason_counters = defaultdict(int)

    for w in tqdm(wcaptions, desc="Building triplets"):
        pid1 = str(w.get("product_id_1", ""))
        pid2 = str(w.get("product_id_2", pid1))
        ir_path = w.get("Ir_path")

        if pid1 not in all_pids or pid2 not in all_pids:
            skip_q += 1
            reason_counters["pid_not_in_docs"] += 1
            continue

        query = {
            "Ir_path": ir_path,
            "text": str(w.get("caption", "")).strip() or " ",
            "id": pid1,
            "category": w["category_1"]  # ← ref
        }

        pos_doc = pid2doc.get(pid2)
        if pos_doc is None:
            skip_q += 1
            reason_counters["pos_doc_missing"] += 1
            continue

        cand_neg = list(all_pids - {pid1, pid2})
        if not cand_neg:
            skip_q += 1
            reason_counters["no_neg_candidates"] += 1
            continue

        neg_docs = []
        n_neg = min(max_negatives, len(cand_neg))
        if n_neg > 0:
            neg_pids = random.sample(cand_neg, n_neg)
            neg_docs = [pid2doc[pid] for pid in neg_pids if pid in pid2doc]

        data.append((query, pos_doc, neg_docs))

    random.shuffle(data)
    N = len(data)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    logger.info(f"[split] Train={len(train_data)} Val={len(val_data)} Test={len(test_data)}")

    train_dataset = QwenVLTripletDataset(train_data, cache_dir=cache_dir)
    val_dataset = QwenVLTripletDataset(val_data, cache_dir=cache_dir)
    test_dataset = QwenVLTripletDataset(test_data, cache_dir=cache_dir)

    def make_loader(dataset, shuffle, drop_last, is_val_or_test):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda b: triplet_collate_multimodal(b, model=model, device=device_t, cache_dir=cache_dir, all_pids=all_pids, pid2doc=pid2doc, debug=debug, is_val_or_test=is_val_or_test, seed=seed),
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            prefetch_factor=2,
            persistent_workers=(num_workers > 0),
            timeout=60
        )

    train_loader = make_loader(train_dataset, shuffle=True, drop_last=True, is_val_or_test=False)
    val_loader = make_loader(val_dataset, shuffle=False, drop_last=False, is_val_or_test=True)
    test_loader = make_loader(test_dataset, shuffle=False, drop_last=False, is_val_or_test=True)

    return train_loader, val_loader, test_loader, documents