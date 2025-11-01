import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from collections import Counter

def load_tokenizer_and_vocab(data, fashion_attr_dict, max_vocab_size=736):
    # 1. Extract words from captions
    caption_words = []
    for item in data:
        words = item["caption"].strip().split()
        caption_words.extend(words)

    # Count word frequencies
    word_counts = Counter(caption_words)

    # 2. Get fashion attributes
    if isinstance(fashion_attr_dict, dict):
        fashion_tokens = list(fashion_attr_dict.keys())
    else:
        fashion_tokens = list(fashion_attr_dict)

    # 3. Combine and select top words
    # Ensure all fashion attributes are included
    vocab = set(fashion_tokens)
    remaining_slots = max_vocab_size - len(vocab) - 4  # Reserve 4 for special tokens
    # Add most frequent caption words
    frequent_words = [word for word, count in word_counts.most_common() if word not in vocab]
    vocab.update(frequent_words[:remaining_slots])

    # 4. Add special tokens
    special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]
    vocab.update(special_tokens)
    vocab = list(vocab)

    print(f"✅ Built vocabulary with {len(vocab)} tokens (including {len(special_tokens)} special tokens and {len(fashion_tokens)} fashion attributes)")

    # 5. Create custom tokenizer
    tokenizer = Tokenizer(WordLevel(vocab={token: idx for idx, token in enumerate(vocab)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    # 6. Save tokenizer (optional, for debugging)
    tokenizer.save("custom_tokenizer.json")

    # 7. Create token_to_id and id_to_token mappings
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    # 8. Sanity check special token IDs
    print("🔎 Special token IDs:")
    print(f"  BOS: {token_to_id['<s>']} (<s>)")
    print(f"  EOS: {token_to_id['</s>']} (</s>)")
    print(f"  PAD: {token_to_id['<pad>']} (<pad>)")
    print(f"  UNK: {token_to_id['<unk>']} (<unk>)")

    return tokenizer, token_to_id, id_to_token

tokenizer, token_to_id, id_to_token = load_tokenizer_and_vocab(wcaptions, splitted_attr_from_cap, len(from_captions))

from tqdm import tqdm

def retokenize_captions(wcaptions, tokenizer, token_to_id, id_to_token, max_caption_len=35):
    pad_token_id = token_to_id['<pad>']
    bos_token_id = token_to_id['<s>']
    eos_token_id = token_to_id['</s>']
    unk_token_id = token_to_id['<unk>']

    vocab_size = len(token_to_id)

    # Validate vocabulary consistency
    if len(token_to_id) != len(id_to_token):
        raise ValueError(f"Vocabulary size mismatch: token_to_id={len(token_to_id)}, id_to_token={len(id_to_token)}")
    for token, idx in token_to_id.items():
        if idx in id_to_token and id_to_token[idx] != token:
            raise ValueError(f"Mismatch: token {token} maps to ID {idx}, but id_to_token[{idx}] = {id_to_token[idx]}")

    updated_wcaptions = []

    for idx, item in enumerate(tqdm(wcaptions, desc="Retokenizing captions")):
        new_item = item.copy()
        caption = item['caption'].strip().lower()

        # Tokenize using custom WordLevel tokenizer
        encoding = tokenizer.encode(caption)
        caption_ids = [bos_token_id] + encoding.ids[:max_caption_len-2] + [eos_token_id]
        caption_len = len(caption_ids)

        # Check for <unk> tokens
        if unk_token_id in caption_ids:
            print(f"⚠️ <unk> token found in caption at index {idx}: {caption}")

            continue

        # Validate token IDs
        if not all(0 <= id < vocab_size for id in caption_ids):
            print(f"⚠️ Invalid token IDs in caption at index {idx}: {caption_ids}")

            continue

        # Pad or truncate to max_caption_len
        if caption_len < max_caption_len:
            caption_ids += [pad_token_id] * (max_caption_len - caption_len)
        elif caption_len > max_caption_len:
            caption_ids = caption_ids[:max_caption_len]
            caption_len = max_caption_len

        # Create attention mask
        attention_mask = [1] * caption_len + [0] * (max_caption_len - caption_len)

        # Decode caption for verification
        decoded_tokens = [id_to_token[id] for id in caption_ids if id not in [bos_token_id, eos_token_id, pad_token_id]]
        decoded_caption = ' '.join(decoded_tokens).strip()

        new_item['caption_input_ids'] = caption_ids
        new_item['attention_mask'] = attention_mask
        new_item['decoded_caption'] = decoded_caption
        updated_wcaptions.append(new_item)

    print(f"✅ Retokenized {len(updated_wcaptions)}/{len(wcaptions)} captions successfully")

    return updated_wcaptions

wcaptions = retokenize_captions(wcaptions, tokenizer, token_to_id, id_to_token, 20)

tokenizer, token_to_id, id_to_token = load_tokenizer_and_vocab(wcaptions, splitted_attr_from_cap, len(from_captions))
wcaptions = retokenize_captions(wcaptions, tokenizer, token_to_id, id_to_token, 20)

