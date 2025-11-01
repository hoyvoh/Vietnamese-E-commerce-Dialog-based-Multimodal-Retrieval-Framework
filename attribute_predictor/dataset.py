from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import os
import requests
from io import BytesIO

# Preprocessing for EfficientNet
image_size = 224  # Adjust for EfficientNet model (e.g., 380 for B4/B7) 224 380
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class ProductImageDataset(Dataset):
    def __init__(self, df, cache_dir="./image_cache"):
        self.df = df
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        product_id = row["product_id"]
        image_url = row["image_url"]

        try:
            # Check if image is cached locally
            local_path = os.path.join(self.cache_dir, f"{product_id}.jpg")
            if os.path.exists(local_path):
                img = Image.open(local_path).convert("RGB")
            else:
                # Handle local or URL-based images
                if image_url.startswith("http"):
                    res = requests.get(image_url, timeout=5)
                    res.raise_for_status()
                    img = Image.open(BytesIO(res.content)).convert("RGB")
                    img.save(local_path, quality=85)  # Cache locally with reduced quality
                else:
                    img = Image.open(image_url).convert("RGB")

            img_tensor = self.transform(img)
            label_tensor = torch.tensor(row.iloc[2:].astype(float).values, dtype=torch.float32)
            return img_tensor, label_tensor, product_id
        except Exception as e:
            print(f"❌ Error with {product_id} - {image_url}: {e}")
            return None, None, product_id

def load_images_and_labels(df, batch_size=32, num_workers=4, cache_dir="./image_cache"):
    dataset = ProductImageDataset(df, cache_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Faster GPU transfer
        drop_last=False
    )

    image_tensors = []
    label_tensors = []
    product_ids = []

    for batch in tqdm(dataloader, desc="Processing images"):
        img_batch, label_batch, ids = batch
        for img, label, pid in zip(img_batch, label_batch, ids):
            if img is not None and label is not None:
                image_tensors.append(img)
                label_tensors.append(label)
                product_ids.append(pid)

    # Stack only if data exists
    if image_tensors and label_tensors:
        X = torch.stack(image_tensors)
        y = torch.stack(label_tensors)
    else:
        X, y = torch.tensor([]), torch.tensor([])

    return X, y, product_ids


X, y, product_ids = load_images_and_labels(df, batch_size=32, num_workers=4)
