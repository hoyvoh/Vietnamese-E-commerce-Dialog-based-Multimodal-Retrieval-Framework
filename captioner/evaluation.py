from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import torch
import re

def evaluate_model_on_testset(model, test_loader, device, max_len=15, top_p=0.9, temperature=1.2):
    """Evaluate the model on the test set using BLEU, ROUGE-L, CIDEr, and SPICE metrics."""
    model.eval()
    references = []
    hypotheses = []
    rouge_scores = []
    coco_references = {}
    coco_hypotheses = {}

    special_ids = {
        'pad': model.pad_token_id,  # 438
        'bos': model.bos_token_id,  # 3
        'eos': model.eos_token_id   # 432
    }

    def preprocess_caption(caption):
        """Preprocess a caption for evaluation by removing punctuation, lowercasing, and normalizing spaces."""
        if not isinstance(caption, str):
            caption = str(caption)
        caption = caption.lower().strip()
        caption = re.sub(r'[^\w\s]', '', caption)  # Remove punctuation
        caption = re.sub(r'\s+', ' ', caption)  # Normalize spaces
        return caption

    def decode_batch_ids(batch_ids, id_to_token, special_ids):
        """Decode batch of token IDs, excluding special tokens and truncating at eos."""
        decoded_texts = []
        for ids in batch_ids:
            eos_idx = (ids == special_ids['eos']).nonzero(as_tuple=True)[0]
            if len(eos_idx) > 0:
                ids = ids[:eos_idx[0]]
            tokens = []
            for id in ids:
                token = id_to_token.get(id.item(), '<unk>')
                if id.item() not in special_ids.values():
                    if token == '<unk>':
                        print(f"Warning: Encountered <unk> token for ID {id.item()}")
                    tokens.append(token)
            decoded_text = ' '.join(tokens).strip()
            if not decoded_text:
                print(f"Warning: Empty decoded text for IDs: {ids.tolist()}")
            decoded_texts.append(decoded_text)
        return decoded_texts

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    cider_scorer = Cider()
    spice_scorer = Spice()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="🚀 Evaluating on Test Set")):
            try:
                # Extract batch inputs
                Ir = batch['Ir'].to(device)  # [batch_size, 3, 224, 224]
                It = batch['It'].to(device)  # [batch_size, 3, 224, 224]
                captions = batch['caption_text']  # List of strings
                batch_size = Ir.size(0)

                # Generate captions using nucleus sampling
                input_ids = nucleus_sampling_vectorized(
                    model, Ir, It, max_len=max_len, top_p=top_p, device=device, temperature=temperature
                )
                sampled_texts = decode_batch_ids(input_ids, model.id_to_token, special_ids)
                gt_texts = [preprocess_caption(c) for c in captions]

                # Debug: Print first few captions for inspection
                if idx == 0:
                    print(f"Sample GT texts (first 2): {gt_texts[:2]}")
                    print(f"Sample PR texts (first 2): {sampled_texts[:2]}")

                for i in range(batch_size):
                    ref = gt_texts[i]
                    hyp = preprocess_caption(sampled_texts[i])
                    img_id = f"img_{idx * batch_size + i}"  # Generate unique image ID

                    # For BLEU
                    references.append([ref.split()])
                    hypotheses.append(hyp.split())

                    # For ROUGE-L
                    score = scorer.score(ref, hyp)['rougeL'].fmeasure
                    rouge_scores.append(score)

                    # For CIDEr and SPICE
                    coco_references[img_id] = [ref]
                    coco_hypotheses[img_id] = [hyp]

            except Exception as e:
                print(f"Error in test batch {idx+1}: {e}")
                continue

    # Compute BLEU scores
    bleu_scores = {
        'BLEU-1': corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0)) if references else 0.0,
        'BLEU-2': corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0)) if references else 0.0,
        'BLEU-3': corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0)) if references else 0.0,
        'BLEU-4': corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)) if references else 0.0
    }

    # Compute average ROUGE-L
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    # Compute CIDEr and SPICE
    cider_score, _ = cider_scorer.compute_score(coco_references, coco_hypotheses) if coco_references else (0.0, [])
    spice_score, _ = spice_scorer.compute_score(coco_references, coco_hypotheses) if coco_hypotheses else (0.0, [])

    metrics = {
        **bleu_scores,
        'ROUGE-L': avg_rouge_l,
        'CIDEr': cider_score,
        'SPICE': spice_score
    }

    return metrics, references, hypotheses
