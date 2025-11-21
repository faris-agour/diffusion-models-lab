import torch
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

IMG_SIZE   = 64
BATCH_SIZE = 8

def ensure_pil_rgb(im):
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    if torch.is_tensor(im):
        arr = im.detach().cpu().permute(1,2,0).numpy()
        return Image.fromarray((arr*255).astype(np.uint8)).convert("RGB")
    arr = np.array(im)
    return Image.fromarray(arr).convert("RGB")

data_transform = transforms.Compose([
    transforms.Lambda(ensure_pil_rgb),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t*2 - 1),
])

def apply_transform(x):
    imgs = x["image"]
    x["pixel_values"] = data_transform(ensure_pil_rgb(imgs))
    return x

def load_dataloader():
    train = load_dataset("tanganke/stanford_cars", split="train")
    test  = load_dataset("tanganke/stanford_cars", split="test")

    data = concatenate_datasets([train, test])
    data = data.with_transform(apply_transform)

    def collate_fn(batch):
        xs = torch.stack([b["pixel_values"] for b in batch])
        ys = torch.tensor([int(b["label"]) for b in batch])
        return xs, ys

    return DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
