"""
Data Preparation Module for Product Captioner
Combines functionality from prepare_data.py and prepare_image_pairs.py
"""

import json
import os
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import openai
from tqdm import tqdm
import random
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# From prepare_data.py
def load_tokenizer_and_vocab(data, fashion_attr_dict, max_vocab_size=736):
    """
    Load tokenizer and vocabulary from data and attributes
    """
    # Collect all captions
    captions = []
    for item in data:
        captions.append(item["caption"])
    
    # Add attribute vocabulary
    attr_vocab = list(fashion_attr_dict.keys())
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Train tokenizer
    trainer = trainers.WordLevelTrainer(
        vocab_size=max_vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"]
    )
    
    tokenizer.train_from_iterator(captions + attr_vocab, trainer)
    
    # Build vocab mappings
    vocab = tokenizer.get_vocab()
    token_to_id = vocab
    id_to_token = {v: k for k, v in vocab.items()}
    
    return tokenizer, token_to_id, id_to_token, attr_vocab

def retokenize_captions(wcaptions, tokenizer, token_to_id, id_to_token, max_caption_len=35):
    """
    Retokenize captions using the trained tokenizer
    """
    retokenized_data = []
    
    for item in wcaptions:
        caption = item["caption"]
        tokens = tokenizer.encode(caption).tokens
        
        # Convert to IDs
        token_ids = []
        for token in tokens[:max_caption_len-2]:  # Leave space for BOS/EOS
            if token in token_to_id:
                token_ids.append(token_to_id[token])
            else:
                token_ids.append(token_to_id["<unk>"])
        
        # Add BOS and EOS
        final_ids = [token_to_id["<s>"]] + token_ids + [token_to_id["</s>"]]
        
        retokenized_item = item.copy()
        retokenized_item["token_ids"] = final_ids
        retokenized_item["tokens"] = tokens
        retokenized_data.append(retokenized_item)
    
    return retokenized_data

# From prepare_image_pairs.py
class CaptionPair(BaseModel):
    """Product pair with comparative captions"""
    ref_product_id: str
    target_product_id: str
    ref_image_path: str
    target_image_path: str
    ref_attributes: List[str]
    target_attributes: List[str]
    comparative_captions: List[str]
    jaccard_similarity: float

def find_product_pairs(all_label, min_similarity=0.3, min_pairs=3, max_pairs=5, 
                      output_path="/content/drive/MyDrive/Training Drive/fashionIQ/product_pairs_with_images.json"):
    """
    Find similar product pairs based on attribute overlap
    """
    products = list(all_label.keys())
    pairs = []
    
    print(f"Finding pairs from {len(products)} products...")
    
    for i, ref_id in enumerate(tqdm(products, desc="Processing products")):
        ref_attrs = set(all_label[ref_id])
        pair_count = 0
        
        # Find similar products
        candidates = []
        for j, target_id in enumerate(products):
            if i >= j:  # Avoid duplicates and self-pairs
                continue
                
            target_attrs = set(all_label[target_id])
            
            # Calculate Jaccard similarity
            intersection = len(ref_attrs & target_attrs)
            union = len(ref_attrs | target_attrs)
            
            if union == 0:
                continue
                
            jaccard_sim = intersection / union
            
            if jaccard_sim >= min_similarity:
                candidates.append((target_id, jaccard_sim, target_attrs))
        
        # Sort by similarity and take top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for target_id, jaccard_sim, target_attrs in candidates[:max_pairs]:
            if pair_count >= max_pairs:
                break
                
            pair = {
                "ref_product_id": ref_id,
                "target_product_id": target_id,
                "ref_attributes": list(ref_attrs),
                "target_attributes": list(target_attrs),
                "jaccard_similarity": jaccard_sim,
                "comparative_captions": []  # To be filled later
            }
            pairs.append(pair)
            pair_count += 1
    
    print(f"Found {len(pairs)} product pairs")
    
    # Save pairs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
    return pairs

def save_checkpoint(results: List[CaptionPair], output_path: str):
    """Save generation checkpoint"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([item.dict() for item in results], f, indent=2, ensure_ascii=False)

def load_checkpoint(output_path: str) -> List[CaptionPair]:
    """Load generation checkpoint"""
    if not os.path.exists(output_path):
        return []
    
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return [CaptionPair(**item) for item in data]

def generate_comparative_captions_batch(
    pairs_with_corpus, 
    batch_size: int = 4, 
    output_path: str = "/content/drive/MyDrive/Training Drive/fashionIQ/comparative_captions.json",
    model: str = "gpt-4o", 
    max_retry: int = 3
):
    """
    Generate comparative captions for product pairs using LLM
    """
    # Load existing results
    existing_results = load_checkpoint(output_path)
    processed_pairs = {(r.ref_product_id, r.target_product_id) for r in existing_results}
    
    # Filter unprocessed pairs
    unprocessed_pairs = [
        pair for pair in pairs_with_corpus 
        if (pair["ref_product_id"], pair["target_product_id"]) not in processed_pairs
    ]
    
    print(f"Processing {len(unprocessed_pairs)} new pairs...")
    
    # Process in batches
    for i in tqdm(range(0, len(unprocessed_pairs), batch_size), desc="Generating captions"):
        batch = unprocessed_pairs[i:i+batch_size]
        
        for pair in batch:
            for retry in range(max_retry):
                try:
                    # Generate captions using LLM
                    captions = _generate_captions_for_pair(pair, model)
                    
                    # Create result
                    result = CaptionPair(
                        ref_product_id=pair["ref_product_id"],
                        target_product_id=pair["target_product_id"],
                        ref_image_path=pair.get("ref_image_path", ""),
                        target_image_path=pair.get("target_image_path", ""),
                        ref_attributes=pair["ref_attributes"],
                        target_attributes=pair["target_attributes"],
                        comparative_captions=captions,
                        jaccard_similarity=pair["jaccard_similarity"]
                    )
                    
                    existing_results.append(result)
                    break
                    
                except Exception as e:
                    print(f"Error generating captions for pair {pair['ref_product_id']}->{pair['target_product_id']}: {e}")
                    if retry == max_retry - 1:
                        # Add empty result to avoid reprocessing
                        result = CaptionPair(
                            ref_product_id=pair["ref_product_id"],
                            target_product_id=pair["target_product_id"],
                            ref_image_path="",
                            target_image_path="",
                            ref_attributes=pair["ref_attributes"],
                            target_attributes=pair["target_attributes"],
                            comparative_captions=[],
                            jaccard_similarity=pair["jaccard_similarity"]
                        )
                        existing_results.append(result)
        
        # Save checkpoint after each batch
        save_checkpoint(existing_results, output_path)
    
    return existing_results

def _generate_captions_for_pair(pair: Dict[str, Any], model: str = "gpt-4o") -> List[str]:
    """
    Generate comparative captions for a single product pair using LLM
    """
    ref_attrs = pair["ref_attributes"]
    target_attrs = pair["target_attributes"]
    
    # Find differences
    ref_only = set(ref_attrs) - set(target_attrs)
    target_only = set(target_attrs) - set(ref_attrs)
    
    prompt = f"""
    Tạo 3-5 câu mô tả ngắn gọn (≤20 từ) bằng tiếng Việt để so sánh 2 sản phẩm thời trang.
    
    Sản phẩm gốc có: {', '.join(ref_attrs)}
    Sản phẩm đích có: {', '.join(target_attrs)}
    
    Khác biệt chính:
    - Chỉ có ở gốc: {', '.join(ref_only) if ref_only else 'Không có'}
    - Chỉ có ở đích: {', '.join(target_only) if target_only else 'Không có'}
    
    Tạo các câu mô tả kiểu "như cái này nhưng [khác biệt]":
    """
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia thời trang Việt Nam, tạo mô tả sản phẩm ngắn gọn."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract captions (assume each line is a caption)
        captions = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Filter and clean captions
        cleaned_captions = []
        for caption in captions:
            # Remove numbering
            caption = caption.lstrip('0123456789.- ')
            if len(caption.split()) <= 20 and len(caption) > 5:
                cleaned_captions.append(caption)
        
        return cleaned_captions[:5]  # Return max 5 captions
        
    except Exception as e:
        print(f"LLM API error: {e}")
        # Return fallback captions
        return [
            f"như sản phẩm này nhưng khác {random.choice(list(target_only))}",
            f"tương tự nhưng có {random.choice(list(target_only)) if target_only else 'màu khác'}"
        ]

# Utility functions
def prepare_captioner_data(input_dir: str, output_dir: str, max_vocab_size: int = 736):
    """
    Complete data preparation pipeline for captioner
    Combines both tokenization and pair generation
    """
    print("🔧 Starting complete captioner data preparation...")
    
    # Step 1: Load raw data
    print("📂 Loading raw data...")
    with open(os.path.join(input_dir, "wcaptions.json"), 'r', encoding='utf-8') as f:
        wcaptions = json.load(f)
    
    with open(os.path.join(input_dir, "attributes.json"), 'r', encoding='utf-8') as f:
        attributes = json.load(f)
    
    # Step 2: Build tokenizer and vocabulary
    print("🔤 Building tokenizer and vocabulary...")
    tokenizer, token_to_id, id_to_token, attr_vocab = load_tokenizer_and_vocab(
        wcaptions, attributes, max_vocab_size
    )
    
    # Step 3: Retokenize captions
    print("🔄 Retokenizing captions...")
    retokenized_data = retokenize_captions(wcaptions, tokenizer, token_to_id, id_to_token)
    
    # Step 4: Find product pairs
    print("👥 Finding product pairs...")
    pairs = find_product_pairs(attributes, output_path=os.path.join(output_dir, "product_pairs.json"))
    
    # Step 5: Generate comparative captions (optional - requires API key)
    print("💬 Generating comparative captions...")
    try:
        caption_pairs = generate_comparative_captions_batch(
            pairs, 
            output_path=os.path.join(output_dir, "comparative_captions.json")
        )
        print(f"✅ Generated captions for {len(caption_pairs)} pairs")
    except Exception as e:
        print(f"⚠️ Caption generation failed: {e}")
        print("Skipping caption generation - you can run it separately with API key")
    
    # Step 6: Save processed data
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer components
    with open(os.path.join(output_dir, "tokenizer_vocab.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "attr_vocab": attr_vocab
        }, f, indent=2, ensure_ascii=False)
    
    # Save retokenized data
    with open(os.path.join(output_dir, "retokenized_captions.json"), 'w', encoding='utf-8') as f:
        json.dump(retokenized_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Captioner data preparation completed!")
    print(f"📊 Vocabulary size: {len(token_to_id)}")
    print(f"📊 Attribute vocab size: {len(attr_vocab)}")
    print(f"📊 Product pairs: {len(pairs)}")
    print(f"📂 Output saved to: {output_dir}")
    
    return {
        "tokenizer": tokenizer,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "attr_vocab": attr_vocab,
        "retokenized_data": retokenized_data,
        "pairs": pairs
    }
