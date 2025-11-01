import json
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import os

def find_product_pairs(all_label, min_similarity=0.3, min_pairs=3, max_pairs=5, output_path="/content/drive/MyDrive/Training Drive/fashionIQ/product_pairs_with_images.json"):
    """
    Find pairs of products with at least 60% shared labels but some differences.
    Each product forms at least 3 and at most 5 pairs. Output includes image paths and labels.

    Args:
        all_label (list): List of (product_id, image_path, labels) tuples
        min_similarity (float): Minimum Jaccard similarity (default: 0.6)
        min_pairs (int): Minimum number of pairs per product (default: 3)
        max_pairs (int): Maximum number of pairs per product (default: 5)
        output_path (str): Path to save the pairs as JSON

    Returns:
        list: List of (product_id_1, images_1, labels_1, product_id_2, images_2, labels_2) tuples
    """
    # Step 1: Group labels and images by product_id
    product_info = defaultdict(lambda: {"labels": set(), "images": set()})
    for product_id, image_path, labels in all_label:
        product_info[product_id]["labels"].update(labels)
        product_info[product_id]["images"].add(image_path)

    # Convert image sets to sorted lists for consistency
    for pid in product_info:
        product_info[pid]["images"] = sorted(list(product_info[pid]["images"]))

    print(f"✅ Grouped {len(all_label)} images into {len(product_info)} products")

    # Step 2: Compute Jaccard similarity for all product pairs
    product_pairs = []
    product_ids = list(product_info.keys())

    for i, pid1 in enumerate(tqdm(product_ids, desc="Computing similarities")):
        labels1 = product_info[pid1]["labels"]
        candidate_pairs = []

        for pid2 in product_ids[i+1:]:  # Avoid duplicate pairs
            labels2 = product_info[pid2]["labels"]
            if labels1 == labels2:  # Skip identical label sets
                continue

            # Calculate Jaccard similarity
            intersection = len(labels1 & labels2)
            union = len(labels1 | labels2)
            similarity = intersection / union if union > 0 else 0

            if similarity >= min_similarity:
                candidate_pairs.append((pid2, similarity))

        # Sort candidates by similarity (descending) and select top min_pairs to max_pairs
        candidate_pairs.sort(key=lambda x: x[1], reverse=True)
        selected_pairs = candidate_pairs[:max_pairs]

        # If fewer than min_pairs, warn but proceed
        if len(selected_pairs) < min_pairs:
            print(f"⚠️ Product {pid1} has only {len(selected_pairs)} pairs (minimum: {min_pairs})")

        # Add pairs to result with image paths and labels
        for pid2, _ in selected_pairs:
            product_pairs.append((
                pid1,
                product_info[pid1]["images"],
                list(product_info[pid1]["labels"]),
                pid2,
                product_info[pid2]["images"],
                list(product_info[pid2]["labels"])
            ))

    print(f"✅ Found {len(product_pairs)} product pairs")

    # Step 3: Save pairs to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert sets to lists for JSON serialization
    json_pairs = [
        (pid1, images1, labels1, pid2, images2, labels2)
        for pid1, images1, labels1, pid2, images2, labels2 in product_pairs
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_pairs, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved product pairs to {output_path}")

    return product_pairs

# Example usage
all_pairs = find_product_pairs(all_labels, min_similarity=0.3, min_pairs=1, max_pairs=4)
print(f"Sample pairs: {all_pairs[:2]}")

pairs_with_corpus = []

for pair in all_pairs:
    product_id_1, images_1, labels_1, product_id_2, images_2, labels_2 = pair
    product_1 = product_index[product_id_1]
    product_2 = product_index[product_id_2]

    corpus_1 = product_1["corpus"]
    corpus_2 = product_2["corpus"]
    pairs_with_corpus.append([
        product_id_1,
        images_1,
        labels_1,
        corpus_1,
        product_id_2,
        images_2,
        labels_2,
        corpus_2
    ])
pairs_with_corpus[:2]


import json
import os
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path
import time
from google.colab import userdata
from pydantic import BaseModel, Field
from typing import List
import uuid

# === API Key ===
client = OpenAI(api_key=userdata.get("OPENAI_API_KEY"))

# === Label mapping ===
LABELS_DESC = {
    0: "irrelevant",        # không liên quan
    1: "size",              # kích thước (S, M, L, 36, free size, ...)
    2: "color",             # màu sắc (đen, trắng, đỏ đô, ...)
    3: "category",          # danh mục (váy, áo, đầm, set, ...)
    4: "brand",             # thương hiệu
    5: "style",             # phong cách (công sở, sexy, basic, thể thao, ...)
    6: "material",          # chất liệu (lụa, thun lạnh, denim, cotton,...)
    7: "silhouette",        # dáng / form (xòe, ôm, suông, croptop, ...)
    8: "detail",            # chi tiết (ren, bèo, dây rút, nơ, nút, ...)
    9: "pattern",           # họa tiết (hoa, kẻ sọc, chấm bi, in chữ, ...)
    10: "target",           # đối tượng (trẻ em, trung niên, bà bầu, ...)
    11: "part",             # bộ phận cơ thể nhắm đến (9eo, ngực, lưng hở, ...)
    12: "usage"             # dịp sử dụng / mục đích (dự tiệc, đi biển, đi làm, ngủ,...)
}

# === Pydantic schema ===
class CaptionPair(BaseModel):
    product_id_1: int
    labels_1: List[str]
    corpus_1: str
    product_id_2: int
    labels_2: List[str]
    corpus_2: str
    captions: List[str] = Field(min_items=0, description="List of comparative captions (min 5 if valid)")

# === Function definition for OpenAI API ===
functions = [
    {
        "name": "generate_comparative_captions",
        "description": "Generate comparative captions for pairs of fashion products based on corpus and labels.",
        "parameters": {
            "type": "object",
            "properties": {
                "caption_pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pair": {"type": "integer", "description": "Pair index (1-based)"},
                            "captions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 5,
                                "description": "List of at least 5 comparative captions in Vietnamese"
                            }
                        },
                        "required": ["pair", "captions"]
                    },
                    "description": "List of caption sets for each pair"
                }
            },
            "required": ["caption_pairs"]
        }
    }
]

def save_checkpoint(results: List[CaptionPair], output_path: str):
    """Save partial results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.dict() for r in results], f, indent=4, ensure_ascii=False)
    print(f"✅ Checkpoint saved: {len(results)} pairs to {output_path}")

def load_checkpoint(output_path: str) -> List[CaptionPair]:
    """Load existing results from JSON."""
    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [CaptionPair(**item) for item in data]
    return []

def generate_comparative_captions_batch(pairs_with_corpus, batch_size: int = 4, output_path: str = "/content/drive/MyDrive/Training Drive/fashionIQ/comparative_captions.json", model: str = "gpt-4o", max_retry: int = 3):
    """
    Generate 5-8 comparative captions for each product pair in batches using OpenAI API based on corpus and labels.
    Returns structured output using Pydantic with checkpointing.

    Args:
        pairs_with_corpus (list): List of (product_id_1, images_1, labels_1, corpus_1, product_id_2, images_2, labels_2, corpus_2) tuples
        batch_size (int): Number of pairs per batch (e.g., 4 or 8)
        output_path (str): Path to save the captions as JSON
        model (str): OpenAI model to use (default: gpt-4o)
        max_retry (int): Maximum retry attempts for API calls

    Returns:
        List[CaptionPair]: List of validated caption pair objects
    """
    # Load existing results for resuming
    results = load_checkpoint(output_path)
    processed_pairs = {(r.product_id_1, r.product_id_2) for r in results}
    remaining_pairs = [p for p in pairs_with_corpus if (p[0], p[4]) not in processed_pairs]

    # Revised example prompts to clarify Product 1 (reference) and Product 2 (target)
    example_prompt = """
**Example 1**:
- **Reference Product (Product 1)**:
  - Corpus: Áo thun trắng in hình hà mã, dáng ôm, cổ cao, tay dài, phong cách basic.
  - Labels: áo, trắng, hà mã, dáng ôm, cổ cao, tay dài, basic
- **Target Product (Product 2)**:
  - Corpus: Áo thun xanh form rộng, cổ tròn, tay lỡ, in hình con mèo, phong cách năng động.
  - Labels: áo thun, xanh, form rộng, cổ tròn, tay lỡ, con mèo, năng động
Example captions (describing how Product 2 differs from Product 1, guiding a salesperson):
[
    "Tôi cần áo thun màu xanh với họa tiết con mèo thay vì hà mã như áo hiện tại.",
    "Sản phẩm tôi tìm là áo thun xanh, form rộng, cổ tròn và tay lỡ, phong cách năng động.",
    "Tôi muốn áo thun có màu xanh, form rộng hơn, không phải dáng ôm như áo này.",
    "Cần tìm áo thun xanh với hình con mèo, cổ tròn, khác với áo trắng cổ cao này.",
    "Tôi đang tìm áo thun năng động, màu xanh, tay lỡ, không phải kiểu basic như áo này.",
    "Sản phẩm mong muốn là áo thun xanh, form rộng, có họa tiết con mèo, thay vì trắng dáng ôm."
]

**Example 2**:
- **Reference Product (Product 1)**:
  - Corpus: Đầm suông màu xanh, phong cách công sở, kích cỡ freesize, dành cho nữ, kiểu Hàn Quốc.
  - Labels: freesize, công_sở, xanh, đầm, đầm_suông, hàn_quốc, nữ
- **Target Product (Product 2)**:
  - Corpus: Đầm suông màu nâu, phong cách công sở, kích cỡ 3xl, dành cho nữ, kiểu Hàn Quốc.
  - Labels: 3xl, công_sở, nâu, đầm, hàn_quốc, nữ, đầm_suông
Example captions (describing how Product 2 differs from Product 1, guiding a salesperson):
[
    "Tôi cần đầm nữ màu nâu thay vì màu xanh như đầm này.",
    "Sản phẩm tôi tìm là đầm suông nâu, kích cỡ 3xl, phù hợp phong cách công sở.",
    "Cần tìm đầm công sở màu nâu, kích cỡ lớn hơn như 3xl so với freesize.",
    "Tôi muốn đầm nữ màu nâu, vẫn giữ phong cách Hàn Quốc như đầm này.",
    "Sản phẩm mong muốn là đầm suông nâu, kích cỡ 3xl, khác với màu xanh hiện tại.",
    "Tôi đang tìm đầm công sở màu nâu, kích cỡ 3xl, thay vì freesize như đầm này."
]
"""

    additional_instructions = """
You are a fashion expert tasked with generating comparative captions in Vietnamese for pairs of fashion products. For each pair:
- **Product 1** is the **reference product** (the product currently being seen).
- **Product 2** is the **target product** (the product being sought).
- Your task is to generate **5-8 concise captions** in Vietnamese that describe **how Product 2 differs from Product 1**, based on their corpus and labels.
- Focus on key attributes defined in the label mapping:
  - 1: size (e.g., s, m, l, freesize, xl)
  - 2: color (e.g., xanh, nâu, trắng)
  - 3: category (e.g., đầm, áo, quần)
  - 5: style (e.g., công_sở, tiệc, hàn_quốc)
  - 7: silhouette (e.g., suông, ôm, xòe)
  - 9: pattern (e.g., hoa, đồng, kẻ sọc)
  - 10: target (e.g., nữ, nam, bà bầu)
  - 12: usage (e.g., công_sở, tiệc, đi biển)
- Captions should guide a salesperson to understand the differences and select a product matching the target (Product 2).
- Captions should be concise (1-2 sentences), descriptive, and follow the style of the examples above.
- Ensure captions are in Vietnamese and output in JSON format via the `generate_comparative_captions` function call.
- Do not rely on images; use only the provided corpus and labels for generating captions.
- The length of any caption should be no more than 25 words.
"""

    # Process remaining pairs in batches
    for i in tqdm(range(0, len(remaining_pairs), batch_size), desc="Processing batches"):
        batch = remaining_pairs[i:i + batch_size]
        prompt_parts = []
        batch_results = []

        # Prepare prompt for the batch
        for idx, pair in enumerate(batch):
            product_id_1, _, labels_1, corpus_1, product_id_2, _, labels_2, corpus_2 = pair

            # Add to prompt
            prompt_parts.append(
                f"""
**Pair {idx + 1}**:
**Reference Product (ID: {product_id_1})**:
- Corpus: {corpus_1}
- Labels: {', '.join(labels_1)}

**Target Product (ID: {product_id_2})**:
- Corpus: {corpus_2}
- Labels: {', '.join(labels_2)}
"""
            )

            # Initialize result for this pair
            batch_results.append({
                "product_id_1": product_id_1,
                "labels_1": labels_1,
                "corpus_1": corpus_1,
                "product_id_2": product_id_2,
                "labels_2": labels_2,
                "corpus_2": corpus_2,
                "captions": []
            })

        if not prompt_parts:
            results.extend([CaptionPair(**r) for r in batch_results])
            save_checkpoint(results, output_path)
            continue

        # Construct full prompt
        prompt = f"""
You are a fashion expert tasked with generating comparative captions for pairs of fashion products. For each pair, Product 1 is the reference product (the one currently being seen), and Product 2 is the target product (the one being sought). Your task is to generate 5-8 concise captions in Vietnamese that describe how Product 2 differs from Product 1, based on their corpus and labels.

{example_prompt}

{additional_instructions}

{"".join(prompt_parts)}

Provide the captions in a JSON format via the `generate_comparative_captions` function call, ensuring 5-8 captions per pair.
"""

        # Prepare API request
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        for attempt in range(max_retry):
            try:
                # Call OpenAI API
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    functions=functions,
                    function_call={"name": "generate_comparative_captions"},
                    max_tokens=4500,
                    temperature=0.7
                )

                # Extract and parse captions
                captions_response = response.choices[0].message.function_call.arguments
                try:
                    captions_data = json.loads(captions_response).get("caption_pairs", [])
                    print(f"API response (batch {i//batch_size + 1}): {captions_data}")
                except json.JSONDecodeError as e:
                    print(f"⚠️ Failed to parse JSON response for batch {i//batch_size + 1}: {e}")
                    captions_data = []

                # Assign captions to pairs
                for pair_data in captions_data:
                    pair_idx = pair_data["pair"] - 1
                    if 0 <= pair_idx < len(batch_results):
                        captions = pair_data.get("captions", [])
                        if len(captions) < 5:
                            print(f"⚠️ Pair {pair_idx + 1} ({batch_results[pair_idx]['product_id_1']}, {batch_results[pair_idx]['product_id_2']}): Only {len(captions)} captions generated")
                            # Fallback: Generate placeholder captions
                            captions.extend([f"Sản phẩm tôi cần tìm có đặc điểm khác (placeholder {j+1})." for j in range(len(captions), 5)])
                        batch_results[pair_idx]["captions"] = captions

                # Validate with Pydantic
                for result in batch_results:
                    try:
                        validated = CaptionPair(**result)
                        results.append(validated)
                    except Exception as e:
                        print(f"⚠️ Skipped invalid result for pair ({result['product_id_1']}, {result['product_id_2']}): {e}")
                        results.append(CaptionPair(
                            product_id_1=result["product_id_1"],
                            labels_1=result["labels_1"],
                            corpus_1=result["corpus_1"],
                            product_id_2=result["product_id_2"],
                            labels_2=result["labels_2"],
                            corpus_2=result["corpus_2"],
                            captions=[f"Sản phẩm tôi cần tìm có đặc điểm khác (placeholder {j+1})." for j in range(5)]
                        ))

                break  # Exit retry loop on success

            except Exception as e:
                print(f"❌ Retry {attempt+1}/{max_retry} for batch {i//batch_size + 1}: {e}")
                time.sleep(1)
                if attempt == max_retry - 1:
                    print(f"❌ Failed to process batch {i//batch_size + 1} after {max_retry} attempts")
                    results.extend([CaptionPair(
                        product_id_1=r["product_id_1"],
                        labels_1=r["labels_1"],
                        corpus_1=r["corpus_1"],
                        product_id_2=r["product_id_2"],
                        labels_2=r["labels_2"],
                        corpus_2=r["corpus_2"],
                        captions=[f"Sản phẩm tôi cần tìm có đặc điểm khác (placeholder {j+1})." for j in range(5)]
                    ) for r in batch_results])

        # Save checkpoint after each batch
        save_checkpoint(results, output_path)

        # Respect API rate limits
        time.sleep(1)

    print(f"✅ Saved {len(results)} comparative captions to {output_path}")
    return results



