import os
import torch
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from tqdm import tqdm
import json
from sklearn.metrics import ndcg_score
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === METRICS ===
def calculate_mrr(ranks: List[int]) -> float:
    if not ranks: return 0.0
    return np.mean([1.0 / rank for rank in ranks if rank > 0])

def calculate_recall(ranks: List[int], k: int) -> float:
    if not ranks: return 0.0
    return sum(1 for rank in ranks if 0 < rank <= k) / len(ranks)

def calculate_ndcg(relevances: List[List[int]], k: int) -> float:
    if not relevances: return 0.0
    return np.mean([ndcg_score([rel[:k]], [1 if i == 0 else 0]) for rel in relevances])

# === HỖ TRỢ: Write JSONL ===
def _write_jsonl(path: str, payloads: List[Dict[str, Any]]):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for payload in payloads:
                if isinstance(payload, dict):
                    payload["cwd"] = os.getcwd()
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Cannot write jsonl to {path}: {e}")

# === LATE INTERACTION SCORING (INTERNAL) ===
def _evaluate_with_late_interaction_scoring(
    model, queries: List[Dict], index, namespaces: List[str], ks: List[int]
) -> Dict:
    results = {ns: {"ranks": [], "relevances": []} for ns in namespaces}
    max_k = max(ks)

    for query in tqdm(queries, desc="Late Interaction Eval"):
        target_id = str(query["product_id_2"])
        image_path = query["Ir_path"]
        caption = str(query.get("caption", "")).strip() or " "
        target_category = str(query.get("category_2", "other"))

        if not os.path.exists(image_path):
            for ns in namespaces:
                results[ns]["ranks"].append(0)
                results[ns]["relevances"].append([0] * max_k)
            continue

        try:
            # Encode with tokens
            emb_both = model.encode(image_path, caption, mode="both", return_tokens="both")
            emb_image = model.encode(image_path, "", mode="image", return_tokens="both")
            emb_text = model.encode("", caption, mode="text", return_tokens="both")
            query_embs = {"both": emb_both, "image": emb_image, "text": emb_text}
        except Exception as e:
            logger.error(f"Late encoding failed: {e}")
            for ns in namespaces:
                results[ns]["ranks"].append(0)
                results[ns]["relevances"].append([0] * max_k)
            continue

        for ns in namespaces:
            if ns not in query_embs:
                continue
            try:
                q_tokens = query_embs[ns]["tokens"].unsqueeze(0)
                q_mask = query_embs[ns]["attention_mask"].unsqueeze(0)

                # Fetch all vectors in namespace
                all_ids = index.list(namespace=ns)
                if not all_ids:
                    results[ns]["ranks"].append(0)
                    results[ns]["relevances"].append([0] * max_k)
                    continue

                # Batch fetch
                scores = []
                ids = []
                batch_size = 100
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i:i + batch_size]
                    batch_res = index.fetch(ids=batch_ids, namespace=ns)
                    for vec_id, vec in batch_res.vectors.items():
                        doc_tokens = torch.tensor(vec.values).unsqueeze(0)
                        doc_mask = torch.ones(1, device=doc_tokens.device)
                        score = model.compute_late_interaction_score(q_tokens, doc_tokens, q_mask, doc_mask).item()
                        scores.append(score)
                        ids.append(vec_id.replace(f"_{ns}", ""))

                # Sort and rank
                sorted_idx = np.argsort(scores)[::-1]
                top_ids = [ids[i] for i in sorted_idx[:max_k]]
                rank = top_ids.index(target_id) + 1 if target_id in top_ids else 0
                relevance = [1 if pid == target_id else 0 for pid in top_ids]

                results[ns]["ranks"].append(rank)
                results[ns]["relevances"].append(relevance + [0] * (max_k - len(relevance)))
            except Exception as e:
                logger.error(f"Late query failed for {ns}: {e}")
                results[ns]["ranks"].append(0)
                results[ns]["relevances"].append([0] * max_k)

    # Compute metrics
    metrics = {}
    for ns, data in results.items():
        metrics[ns] = {
            "MRR": calculate_mrr(data["ranks"]),
            "Recall": {f"Recall@{k}": calculate_recall(data["ranks"], k) for k in ks},
            "NDCG": {f"NDCG@{k}": calculate_ndcg(data["relevances"], k) for k in ks}
        }
    return metrics

# === POOLED EMBEDDING EVAL (INTERNAL) ===
def _evaluate_with_pooled_embeddings(
    model, queries: List[Dict], index, namespaces: List[str], ks: List[int],
    use_caption: bool, filter_category: bool
) -> Dict:
    results = {ns: {"ranks": [], "relevances": []} for ns in namespaces}
    max_k = max(ks)

    for query in tqdm(queries, desc="Pooled Eval"):
        target_id = str(query["product_id_2"])
        image_path = query["Ir_path"]
        caption = str(query.get("caption", "")).strip() or " "
        target_category = str(query.get("category_2", "other"))

        if not os.path.exists(image_path):
            for ns in namespaces:
                results[ns]["ranks"].append(0)
                results[ns]["relevances"].append([0] * max_k)
            continue

        try:
            emb_both = model.encode(image_path, caption, mode="both").cpu().numpy()
            emb_image = model.encode(image_path, "", mode="image").cpu().numpy()
            emb_text = model.encode("", caption, mode="text").cpu().numpy()
            query_embs = {"both": emb_both, "image": emb_image, "text": emb_text}
        except Exception as e:
            logger.error(f"Pooled encoding failed: {e}")
            for ns in namespaces:
                results[ns]["ranks"].append(0)
                results[ns]["relevances"].append([0] * max_k)
            continue

        for ns in namespaces:
            if ns not in query_embs:
                continue
            try:
                filter_dict = {"category": {"$eq": target_category}} if filter_category else None
                res = index.query(
                    vector=query_embs[ns].tolist(),
                    top_k=max_k,
                    include_metadata=True,
                    namespace=ns,
                    filter=filter_dict
                )
                top_ids = [m["id"].replace(f"_{ns}", "") for m in res["matches"]]
                rank = top_ids.index(target_id) + 1 if target_id in top_ids else 0
                relevance = [1 if pid == target_id else 0 for pid in top_ids]

                results[ns]["ranks"].append(rank)
                results[ns]["relevances"].append(relevance + [0] * (max_k - len(relevance)))
            except Exception as e:
                logger.error(f"Pooled query failed for {ns}: {e}")
                results[ns]["ranks"].append(0)
                results[ns]["relevances"].append([0] * max_k)

    metrics = {}
    for ns, data in results.items():
        metrics[ns] = {
            "MRR": calculate_mrr(data["ranks"]),
            "Recall": {f"Recall@{k}": calculate_recall(data["ranks"], k) for k in ks},
            "NDCG": {f"NDCG@{k}": calculate_ndcg(data["relevances"], k) for k in ks}
        }
    return metrics

# === MAIN EVALUATE FUNCTION ===
def evaluate_search_results(
    model,
    queries: List[Dict[str, Any]],
    index,
    namespaces: List[str] = ["both", "image", "text"],
    ks: List[int] = [1, 5, 10, 20],
    use_caption: bool = True,
    filter_category: bool = True,
    use_late_interaction: bool = False
) -> Dict[str, Dict[str, float]]:
    if use_late_interaction and hasattr(model, 'enable_late_interaction') and model.enable_late_interaction:
        logger.info("Using LATE INTERACTION scoring")
        return _evaluate_with_late_interaction_scoring(model, queries, index, namespaces, ks)
    else:
        logger.info("Using POOLED EMBEDDING scoring")
        return _evaluate_with_pooled_embeddings(model, queries, index, namespaces, ks, use_caption, filter_category)

# === DIALOG-BASED EVALUATION ===
def evaluate_dialog_success(
    model, queries: List[Dict], index, max_turns: int = 5, namespace: str = "both"
) -> Dict[str, float]:
    """
    Dialog@≤T và Mean Turns như trong CLaMR paper
    """
    success_at_turn = {t: 0 for t in range(1, max_turns + 1)}
    total_turns = []

    for query in tqdm(queries, desc="Dialog Eval"):
        current_turn = 1
        target_id = str(query["product_id_2"])
        image_path = query["Ir_path"]
        current_caption = str(query.get("caption", "")).strip() or " "
        found = False

        while current_turn <= max_turns and not found:
            try:
                emb = model.encode(image_path, current_caption, mode="both").cpu().numpy()
                res = index.query(
                    vector=emb.tolist(),
                    top_k=1,
                    include_metadata=True,
                    namespace=namespace
                )
                if res["matches"] and res["matches"][0]["id"].replace(f"_{namespace}", "") == target_id:
                    success_at_turn[current_turn] += 1
                    total_turns.append(current_turn)
                    found = True
                else:
                    # Simulate refinement (in real system: use LLM)
                    current_caption = f"{current_caption} more specific"
                    current_turn += 1
            except Exception as e:
                current_turn += 1

        if not found:
            total_turns.append(max_turns + 1)

    total_queries = len(queries)
    metrics = {}
    for t in range(1, max_turns + 1):
        cumulative = sum(success_at_turn[i] for i in range(1, t + 1))
        metrics[f"Dialog@≤{t}"] = cumulative / total_queries
    metrics["mean_turns"] = np.mean([min(t, max_turns) for t in total_turns])

    return metrics

# === MODALITY-SPECIFIC ACCURACY ===
def infer_target_modality(caption: str) -> str:
    caption = caption.lower()
    if any(k in caption for k in ["video", "clip", "motion"]):
        return "video"
    if any(k in caption for k in ["sound", "audio", "voice"]):
        return "audio"
    if any(k in caption for k in ["text", "letter", "word"]):
        return "ocr"
    return "metadata"

def test_modality_retrieval(model, query: Dict, index, modality: str, namespace: str = "both") -> bool:
    image_path = query["Ir_path"]
    caption = query["caption"]
    target_id = str(query["product_id_2"])

    try:
        if modality == "video":
            emb = model.encode(image_path, "", mode="image")
        elif modality == "audio":
            emb = model.encode("", caption, mode="text")
        elif modality == "ocr":
            emb = model.encode("", caption, mode="text")
        else:
            emb = model.encode(image_path, caption, mode="both")

        res = index.query(vector=emb.cpu().numpy().tolist(), top_k=1, namespace=namespace)
        return res["matches"] and res["matches"][0]["id"].replace(f"_{namespace}", "") == target_id
    except:
        return False

def evaluate_modality_accuracy(model, queries: List[Dict], index, namespace: str = "both") -> Dict[str, float]:
    modality_results = {
        "video": {"correct": 0, "total": 0},
        "audio": {"correct": 0, "total": 0},
        "ocr": {"correct": 0, "total": 0},
        "metadata": {"correct": 0, "total": 0}
    }

    for query in queries:
        caption = str(query.get("caption", "")).strip()
        if not caption:
            continue
        target_modality = infer_target_modality(caption)
        modality_results[target_modality]["total"] += 1
        if test_modality_retrieval(model, query, index, target_modality, namespace):
            modality_results[target_modality]["correct"] += 1

    accuracy = {}
    for mod, res in modality_results.items():
        accuracy[mod] = res["correct"] / max(1, res["total"])
    return accuracy

# === PRINT METRICS ===
def print_metrics(metrics: Dict, title: str = "EVALUATION RESULTS"):
    print("\n" + "="*70)
    print(f"{'':>20}{title}")
    print("="*70)
    for mode in metrics.keys():
        print(f"\n{mode.upper():<12} MODE:")
        m = metrics[mode]
        if "MRR" in m:
            print(f"  MRR          : {m['MRR']:.4f}")
        for k in [1, 5, 10, 20]:
            if f"Recall@{k}" in m["Recall"]:
                print(f"  Recall@{k:<2}   : {m['Recall'][f'Recall@{k}']:.4f} | NDCG@{k:<2} : {m['NDCG'][f'NDCG@{k}']:.4f}")
        for k in m.keys():
            if k.startswith("Dialog"):
                print(f"  {k} : {m[k]:.4f}")
        if "mean_turns" in m:
            print(f"  Mean Turns   : {m['mean_turns']:.2f}")
    print("="*70)

# === MAIN ===
if __name__ == "__main__":
    # === LOAD MODEL ===
    model = QwenVLTripletEncoder(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        out_dim=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
        debug=False,
        min_pixels=64*28*28,
        max_pixels=896*28*28,
        dtype=torch.bfloat16,
        use_lora=False,
        enable_late_interaction=True,
        late_interaction_mode="context"
    )
    model.requires_grad_(False)
    model_dir = "/content/drive/MyDrive/Training Drive/fashionIQ/checkpoints/dialog_b4/vietnamese_ecommercee"
    model.load_model(model_dir)
    logger.info(f"Model loaded from {model_dir}")

    # === PINECONE ===
    from google.colab import userdata
    pc = Pinecone(api_key=userdata.get("PINECONE_API_KEY"))
    index = pc.Index("qwen-product-index")

    # === SAMPLE QUERIES ===
    sample_queries = random.sample(wcaptions, k=min(1000, len(wcaptions)))

    # === 1. STANDARD EVAL ===
    metrics_pooled = evaluate_search_results(
        model=model, queries=sample_queries, index=index,
        use_late_interaction=False
    )
    print_metrics(metrics_pooled, "POOLED EMBEDDING EVAL")

    # === 2. LATE INTERACTION EVAL ===
    metrics_late = evaluate_search_results(
        model=model, queries=sample_queries, index=index,
        use_late_interaction=True
    )
    print_metrics(metrics_late, "LATE INTERACTION EVAL")

    # === 3. DIALOG EVAL ===
    dialog_metrics = evaluate_dialog_success(model, sample_queries[:100], index)
    print_metrics(dialog_metrics, "DIALOG RETRIEVAL EVAL")

    # === 4. MODALITY ACCURACY ===
    mod_acc = evaluate_modality_accuracy(model, sample_queries[:200], index)
    print("\nMODALITY ACCURACY:")
    for mod, acc in mod_acc.items():
        print(f"  {mod.capitalize():<10}: {acc:.4f}")

    # === SAVE RESULTS ===
    all_results = {
        "pooled": metrics_pooled,
        "late_interaction": metrics_late,
        "dialog": dialog_metrics,
        "modality_accuracy": mod_acc
    }
    result_path = "/content/eval_full_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"All results saved to {result_path}")