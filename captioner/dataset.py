from torch.utils.data import Dataset
from PIL import Image
import torch
from tokenizers import Tokenizer

class CaptioningDataset(Dataset):
    def __init__(self, data, tokenizer, token_to_id, id_to_token, attr_vocab, expected_captions=None,
                 train_transform=None, val_transform=None, max_caption_len=35, min_freq=1, vocab_size_limit = 736):
        self.data = data
        self.tokenizer = tokenizer  # Custom WordLevel tokenizer
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.attr_vocab = attr_vocab
        self.expected_captions = expected_captions
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.max_caption_len = max_caption_len
        self.min_freq = min_freq

        # Ensure special tokens match ProductCaptioningModel
        required_specials = ['<pad>', '<s>', '</s>', '<unk>']
        for token in required_specials:
            if token not in self.token_to_id:
                print(f"Adding missing special token '{token}' to token_to_id")
                self.token_to_id[token] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = token

        # Add frequent words and attributes
        word_counts = {}
        for item in data:
            words = item["caption"].strip().lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        if expected_captions:
            for caption in expected_captions:
                words = caption.strip().lower().split()
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1


        added_tokens = 0
        for word in sorted(word_counts, key=word_counts.get, reverse=True):
            if word not in self.token_to_id and word_counts[word] >= min_freq and len(self.token_to_id) < vocab_size_limit:
                self.token_to_id[word] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = word
                added_tokens += 1
        for attr in attr_vocab:
            if attr not in self.token_to_id and len(self.token_to_id) < vocab_size_limit:
                self.token_to_id[attr] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = attr
                added_tokens += 1
        print(f"Added {added_tokens} new tokens (words and attributes) to vocabulary")

        # Truncate vocabulary if exceeded
        if len(self.token_to_id) > vocab_size_limit:
            print(f"Warning: Vocabulary size {len(self.token_to_id)} exceeds {vocab_size_limit}. Truncating.")
            self.token_to_id = {k: v for k, v in list(self.token_to_id.items())[:vocab_size_limit]}
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # Set special token IDs
        self.pad_token_id = self.token_to_id['<pad>']
        self.bos_token_id = self.token_to_id['<s>']
        self.eos_token_id = self.token_to_id['</s>']
        self.unk_token_id = self.token_to_id['<unk>']

        # Debug: Print special tokens and vocab size
        print(f"Special tokens: pad={self.pad_token_id}, bos={self.bos_token_id}, eos={self.eos_token_id}, unk={self.unk_token_id}")
        print(f"Vocabulary size: {len(self.token_to_id)}")
        if self.unk_token_id == 637:
            print("Warning: <unk> token ID is 637, confirming log issue")
        if 367 in self.id_to_token:
            print(f"Token ID 367 maps to: {self.id_to_token[367]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            Ir = Image.open(item["Ir_path"]).convert("RGB")
            It = Image.open(item["It_path"]).convert("RGB")
        except Exception as e:
            print(f"Error loading images for index {idx}: {e}")
            Ir = Image.new("RGB", (224, 224))
            It = Image.new("RGB", (224, 224))

        transform = self.train_transform if self.train_transform else self.val_transform
        Ir = transform(Ir)
        It = transform(It)

        caption = self.expected_captions[idx] if self.expected_captions else item["caption"]
        caption = caption.strip().lower()
        tokens = self.tokenizer.encode(caption).tokens
        caption_ids = [self.bos_token_id]
        for token in tokens[:self.max_caption_len - 2]:
            if token in self.token_to_id:
                caption_ids.append(self.token_to_id[token])
            else:
                print(f"Warning: Token '{token}' not in vocabulary for caption '{caption}' at index {idx}, using <unk>")
                caption_ids.append(self.unk_token_id)
        caption_ids.append(self.eos_token_id)
        caption_len = len(caption_ids)
        if caption_len < self.max_caption_len:
            caption_ids += [self.pad_token_id] * (self.max_caption_len - caption_len)
        caption_input_ids = torch.tensor(caption_ids, dtype=torch.long)
        attention_mask = torch.tensor([1] * caption_len + [0] * (self.max_caption_len - caption_len), dtype=torch.long)

        attr_r = torch.zeros(len(self.attr_vocab), dtype=torch.float)
        attr_t = torch.zeros(len(self.attr_vocab), dtype=torch.float)
        for attr in item["Ir_attributes"]:
            if attr in self.attr_vocab:
                attr_r[self.attr_vocab.index(attr)] = 1.0
            else:
                print(f"Warning: Attribute '{attr}' not in attr_vocab for Ir_attributes at index {idx}")
        for attr in item["It_attributes"]:
            if attr in self.attr_vocab:
                attr_t[self.attr_vocab.index(attr)] = 1.0
            else:
                print(f"Warning: Attribute '{attr}' not in attr_vocab for It_attributes at index {idx}")

        return {
            "Ir": Ir,
            "It": It,
            "caption_input_ids": caption_input_ids,
            "attention_mask": attention_mask,
            "caption_text": item["caption"],
            "attr_r": attr_r,
            "attr_t": attr_t
        }
    