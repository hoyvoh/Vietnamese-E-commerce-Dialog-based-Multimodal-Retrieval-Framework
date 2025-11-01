import torch

def nucleus_sampling_vectorized(model, Ir, It, max_len, top_p, device, temperature=1.2):
    model.eval()
    top_p = float(top_p)
    # print(f"Debug: top_p={top_p}, temperature={temperature}")

    batch_size = Ir.size(0)
    vocab_size = model.vocab_size
    input_ids = torch.full((batch_size, 1), model.bos_token_id, dtype=torch.long, device=device)
    generated = [input_ids]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Get initial context
    with torch.no_grad():
        attr_r, Ir_feat = model.attribute_predictor(Ir, return_features=True)
        attr_t, It_feat = model.attribute_predictor(It, return_features=True)
        Ir_embed = model.visual_embedding(Ir_feat)
        It_embed = model.visual_embedding(It_feat)
        Ar_embed = model.attribute_embedding(attr_r)
        At_embed = model.attribute_embedding(attr_t)
        ref_embed = Ir_embed + model.alpha * Ar_embed
        tgt_embed = It_embed + model.alpha * At_embed
        context = model.joint_encoding(ref_embed, tgt_embed)
        h_0 = context.unsqueeze(0).repeat(model.lstm.num_layers, 1, 1)
        c_0 = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size, device=context.device)

    hidden = (h_0, c_0)
    # print(f"Debug: bos_token_id={model.bos_token_id}, eos_token_id={model.eos_token_id}, pad_token_id={model.pad_token_id}")
    # print(f"Debug: vocab_size={vocab_size}, input_ids.shape={input_ids.shape}")

    with torch.no_grad():
        for step in range(max_len - 1):
            if torch.all(finished):
                # print(f"Debug: All sequences finished at step {step}")
                break
            token_emb = model.token_embedding(input_ids[:, -1:])
            lstm_output, hidden = model.lstm(token_emb, hidden)
            logits = model.output_layer(lstm_output[:, -1, :])

            probs = torch.softmax(logits / temperature, dim=-1)
            # print(f"Debug: Step {step}, probs.shape={probs.shape}, max_prob={probs.max().item():.4f}")

            if top_p == 0.0:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = sorted_probs.cumsum(dim=-1)
                mask = cumulative_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(1, next_token)
                greedy_token = torch.argmax(probs, dim=-1, keepdim=True)
                if torch.equal(next_token, greedy_token):
                    print(f"Warning: Nucleus sampling (top_p={top_p}) produced same token as greedy at step {step}")

            # if (next_token == 367).any():
                # print(f"Debug: Token ID 367 generated at step {step}, count={(next_token == 367).sum().item()}")

            input_ids = torch.where(finished.unsqueeze(1), input_ids[:, -1:], input_ids)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated.append(next_token)
            finished = finished | (next_token.squeeze(1) == model.eos_token_id)
            # print(f"Debug: Step {step}, next_token[0]={next_token[0].item()}, finished={finished.sum().item()}/{batch_size}")

        output = torch.cat(generated, dim=1)
    # print(f"Debug: output.shape={output.shape}, sample_output[0]={output[0].tolist()}")
    return output[:, :max_len]
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import os
import gc
from rouge_score import rouge_scorer

def pretrain_decoder(model, train_loader, device, max_caption_len, pretrain_epochs=5, pretrain_lr=1e-3):
    """
    Pretrain the LSTM decoder.

    Args:
        model: The ProductCaptioningModel instance.
        train_loader: DataLoader for training data.
        device: Device to run the model on (cuda or cpu).
        max_caption_len: Maximum length of captions.
        pretrain_epochs: Number of pretraining epochs (default: 5).
        pretrain_lr: Learning rate for pretraining (default: 1e-3).
    """
    print("Pretraining LSTM decoder...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.lstm.parameters():
        param.requires_grad = True
    for param in model.token_embedding.parameters():
        param.requires_grad = True
    for param in model.output_layer.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=pretrain_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    pretrain_losses = []

    for epoch in range(pretrain_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}"):
            Ir, It, caption_input_ids, _, _, attention_mask = (
                batch['Ir'].to(device), batch['It'].to(device),
                batch['caption_input_ids'].to(device), batch['caption_text'],
                batch['attr_r'].to(device), batch['attr_t'].to(device)
            )
            optimizer.zero_grad()
            try:
                with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    logits, _, _ = model(Ir, It, caption_input_ids, attention_mask)
                    if logits.size(1) != caption_input_ids.size(1) - 1:
                        logits = logits[:, :caption_input_ids.size(1) - 1, :]
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)),
                        caption_input_ids[:, 1:].contiguous().view(-1)
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                # Clean up temporary tensors
                del logits, loss, Ir, It, caption_input_ids, attention_mask
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Error in pretrain batch: {e}")
                continue
        avg_loss = total_loss / len(train_loader)
        pretrain_losses.append(avg_loss)
        print(f"Pretrain Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        # Clean up after each epoch
        torch.cuda.empty_cache()
        gc.collect()

    for param in model.parameters():
        param.requires_grad = True
    del optimizer, criterion, scaler
    torch.cuda.empty_cache()
    gc.collect()
    return pretrain_losses

def decode_batch_ids(batch_ids, id_to_token, special_ids):
    decoded_texts = []
    for ids in batch_ids:
        eos_idx = (ids == special_ids['eos']).nonzero(as_tuple=True)[0]
        if len(eos_idx) > 0:
            ids = ids[:eos_idx[0]]
        tokens = []
        for id in ids:
            if id.item() not in special_ids.values():
                token = id_to_token.get(id.item(), '<unk>')
                if token == '<unk>':
                    print(f"Warning: Encountered <unk> token for id {id.item()}")
                tokens.append(token)
        decoded_text = ' '.join(tokens).strip()
        if not decoded_text:
            print(f"Warning: Empty decoded text for IDs: {ids.tolist()}")
        decoded_texts.append(decoded_text)
    return decoded_texts

def compute_rouge_l(hypotheses, references):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = [scorer.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(references, hypotheses)]
    return sum(scores) / len(scores) if scores else 0.0

def compute_attr_accuracy(pred_attr, true_attr):
    """Compute binary classification accuracy for attribute predictions."""
    pred_binary = (torch.sigmoid(pred_attr) > 0.5).float()
    correct = (pred_binary == true_attr).float().mean().item()
    return correct

def train_product_captioning_model(
    model,
    train_loader,
    val_loader,
    device,
    # Training Hyperparameters
    epochs=10,
    save_path="checkpoints",
    max_caption_len=15,
    debug_one_epoch=False,
    # Pretraining Hyperparameters
    pretrain_epochs=5,
    pretrain_lr=1e-3,
    # Optimization Hyperparameters
    train_lr=5e-5,
    attr_loss_weight=1.0,
    # Sampling Hyperparameters
    top_p=0.9,
    train_temperature=1.5,
    baseline_temperature=1.0
):
    """
    Train the ProductCaptioningModel with tunable hyperparameters and memory cleanup.

    Args:
        model: The ProductCaptioningModel instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to run the model on (cuda or cpu).
        # Training Hyperparameters
        epochs: Number of training epochs (default: 10).
        save_path: Directory to save model checkpoints (default: "checkpoints").
        max_caption_len: Maximum length of generated captions (default: 15).
        debug_one_epoch: Run only one epoch for debugging (default: False).
        # Pretraining Hyperparameters
        pretrain_epochs: Number of pretraining epochs for LSTM decoder (default: 5).
        pretrain_lr: Learning rate for pretraining (default: 1e-3).
        # Optimization Hyperparameters
        train_lr: Learning rate for full model training (default: 5e-5).
        attr_loss_weight: Weight for attribute loss in total loss (default: 1.0).
        # Sampling Hyperparameters
        top_p: Nucleus sampling probability threshold (default: 0.9).
        train_temperature: Temperature for sampling during training (default: 1.5).
        baseline_temperature: Temperature for baseline (greedy) sampling (default: 1.0).
    """
    os.makedirs(save_path, exist_ok=True)
    pretrain_losses = pretrain_decoder(model, train_loader, device, max_caption_len, pretrain_epochs, pretrain_lr)

    optimizer = optim.AdamW(model.parameters(), lr=train_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
    attr_criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    train_losses, val_losses = [], []
    train_rouge_scores, val_rouge_scores = [], []
    train_attr_accs, val_attr_accs = [], []
    best_rouge = 0.0
    best_model_path = os.path.join(save_path, "best_model.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_rouge = 0.0
        total_attr_acc = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            Ir, It, caption_input_ids, caption_text, attr_r, attr_t = (
                batch['Ir'].to(device), batch['It'].to(device),
                batch['caption_input_ids'].to(device), batch['caption_text'],
                batch['attr_r'].to(device), batch['attr_t'].to(device)
            )
            optimizer.zero_grad()
            try:
                with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    logits, pred_attr_r, pred_attr_t = model(Ir, It, caption_input_ids, batch['attention_mask'].to(device))
                    caption_loss = criterion(
                        logits[:, :caption_input_ids.size(1)-1, :].reshape(-1, logits.size(-1)),
                        caption_input_ids[:, 1:].contiguous().view(-1)
                    )
                    attr_loss = (attr_criterion(pred_attr_r, attr_r) + attr_criterion(pred_attr_t, attr_t)) / 2
                    loss = caption_loss + attr_loss_weight * attr_loss
                    attr_acc_r = compute_attr_accuracy(pred_attr_r, attr_r)
                    attr_acc_t = compute_attr_accuracy(pred_attr_t, attr_t)
                    attr_acc = (attr_acc_r + attr_acc_t) / 2
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                total_attr_acc += attr_acc
                # Clean up temporary tensors
                del logits, pred_attr_r, pred_attr_t, loss, caption_loss, attr_loss
            except Exception as e:
                print(f"Error in train batch {batch_idx+1}: {e}")
                continue

            try:
                model.eval()
                with torch.no_grad():
                    sampled_ids = nucleus_sampling_vectorized(
                        model, Ir, It, max_len=max_caption_len, top_p=top_p, device=device, temperature=train_temperature
                    )
                    baseline_ids = nucleus_sampling_vectorized(
                        model, Ir, It, max_len=max_caption_len, top_p=0.0, device=device, temperature=baseline_temperature
                    )
                    sampled_texts = decode_batch_ids(sampled_ids, model.id_to_token, {
                        'pad': model.pad_token_id, 'bos': model.bos_token_id, 'eos': model.eos_token_id
                    })
                    baseline_texts = decode_batch_ids(baseline_ids, model.id_to_token, {
                        'pad': model.pad_token_id, 'bos': model.bos_token_id, 'eos': model.eos_token_id
                    })
                    total_rouge += compute_rouge_l(sampled_texts, caption_text)
                    if not sampled_texts[0] and not baseline_texts[0]:
                        print(f"Warning: Zero reward_diff in batch {batch_idx+1}, sampled_texts == baseline_texts?")
                        print(f"Sampled texts (first 2): {sampled_texts[:2]}")
                        print(f"Baseline texts (first 2): {baseline_texts[:2]}")
                        print(f"Ground truth texts (first 2): {caption_text[:2]}")
                model.train()
                # Clean up temporary tensors
                del sampled_ids, baseline_ids, sampled_texts, baseline_texts, Ir, It, caption_input_ids, caption_text, attr_r, attr_t
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Error in caption generation for batch {batch_idx+1}: {e}")
                model.train()
                continue

        avg_loss = total_loss / len(train_loader)
        avg_rouge = total_rouge / len(train_loader)
        avg_attr_acc = total_attr_acc / len(train_loader)
        train_losses.append(avg_loss)
        train_rouge_scores.append(avg_rouge)
        train_attr_accs.append(avg_attr_acc)
        print(f"Train Epoch {epoch+1} - Loss: {avg_loss:.4f}, ROUGE-L: {avg_rouge:.4f}, Attr Acc: {avg_attr_acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_rouge = 0.0
        val_attr_acc = 0.0
        for batch in tqdm(val_loader, desc=f"Valid Epoch {epoch+1}"):
            Ir, It, caption_input_ids, caption_text, attr_r, attr_t = (
                batch['Ir'].to(device), batch['It'].to(device),
                batch['caption_input_ids'].to(device), batch['caption_text'],
                batch['attr_r'].to(device), batch['attr_t'].to(device)
            )
            try:
                with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    logits, pred_attr_r, pred_attr_t = model(Ir, It, caption_input_ids, batch['attention_mask'].to(device))
                    caption_loss = criterion(
                        logits[:, :caption_input_ids.size(1)-1, :].reshape(-1, logits.size(-1)),
                        caption_input_ids[:, 1:].contiguous().view(-1)
                    )
                    attr_loss = (attr_criterion(pred_attr_r, attr_r) + attr_criterion(pred_attr_t, attr_t)) / 2
                    loss = caption_loss + attr_loss_weight * attr_loss
                    attr_acc_r = compute_attr_accuracy(pred_attr_r, attr_r)
                    attr_acc_t = compute_attr_accuracy(pred_attr_t, attr_t)
                    attr_acc = (attr_acc_r + attr_acc_t) / 2
                val_loss += loss.item()
                val_attr_acc += attr_acc
                sampled_ids = nucleus_sampling_vectorized(
                    model, Ir, It, max_len=max_caption_len, top_p=top_p, device=device, temperature=train_temperature
                )
                sampled_texts = decode_batch_ids(sampled_ids, model.id_to_token, {
                    'pad': model.pad_token_id, 'bos': model.bos_token_id, 'eos': model.eos_token_id
                })
                val_rouge += compute_rouge_l(sampled_texts, caption_text)
                # Clean up temporary tensors
                del logits, pred_attr_r, pred_attr_t, loss, caption_loss, attr_loss, sampled_ids, sampled_texts, Ir, It, caption_input_ids, caption_text, attr_r, attr_t
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
        avg_val_loss = val_loss / len(val_loader)
        avg_val_rouge = val_rouge / len(val_loader)
        avg_val_attr_acc = val_attr_acc / len(val_loader)
        val_losses.append(avg_val_loss)
        val_rouge_scores.append(avg_val_rouge)
        val_attr_accs.append(avg_val_attr_acc)
        print(f"Val Epoch {epoch+1} - Loss: {avg_val_loss:.4f}, ROUGE-L: {avg_val_rouge:.4f}, Attr Acc: {avg_val_attr_acc:.4f}")

        # Save best model based on validation ROUGE-L
        if avg_val_rouge > best_rouge:
            best_rouge = avg_val_rouge
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rouge': avg_val_rouge,
                'val_loss': avg_val_loss,
                'val_attr_acc': avg_val_attr_acc
            }
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model at epoch {epoch+1} with Val ROUGE-L: {avg_val_rouge:.4f} to {best_model_path}")

        # Clean up after each epoch
        torch.cuda.empty_cache()
        gc.collect()

        if debug_one_epoch:
            break

    # Final memory cleanup
    del optimizer, criterion, attr_criterion, scaler
    torch.cuda.empty_cache()
    gc.collect()
    return model, train_losses, val_losses, train_attr_accs, val_attr_accs, train_rouge_scores, val_rouge_scores, pretrain_losses
