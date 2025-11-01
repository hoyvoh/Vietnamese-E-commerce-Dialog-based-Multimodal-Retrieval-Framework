import torch
from tqdm import tqdm
import os

def train_with_early_stopping(model_swin, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10, checkpoint_path="/content/drive/MyDrive/Training Drive/fashionIQ/best_model.pth"):
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        # Training phase
        model_swin.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model_swin(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model_swin.eval()
        total_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model_swin(xb)
                loss = criterion(preds, yb)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Check for improvement
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model_swin.state_dict()
            torch.save(best_model_state, checkpoint_path)
            print(f"✅ Saved best model with Val Loss: {best_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ Patience counter: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
            print(f"Loading best model with Val Loss: {best_loss:.4f}")
            model_swin.load_state_dict(torch.load(checkpoint_path))
            break

    return model_swin
