import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class SAMMedicalDataset(Dataset):
    def __init__(self, data_dir, modality="xray", transform=None):
        self.modality = modality
        self.transform = transform
        self.image_files = []
        self.mask_files = []
        search_path = os.path.join(data_dir, modality)
        if os.path.exists(search_path):
            mask_path = os.path.join(search_path, "masks")
            for root, _, files in os.walk(search_path):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_mask" not in f:
                        img_p = os.path.join(root, f)
                        mask_p = img_p.replace(".png", "_mask.png").replace(".jpg", "_mask.png")
                        if not os.path.exists(mask_p) and os.path.exists(mask_path):
                            mask_p = os.path.join(mask_path, f.replace(".png", "_mask.png"))
                        if os.path.exists(mask_p):
                            self.image_files.append(img_p)
                            self.mask_files.append(mask_p)
        self.image_files.sort()
        print(f"Found {len(self.image_files)} image-mask pairs for modality: {modality}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
        mask = mask.resize((1024, 1024), resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        return image, mask

class SAMProcessor:
    def __init__(self, target_size=1024):
        self.target_size = target_size
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __call__(self, pil_image):
        image = pil_image.resize((self.target_size, self.target_size), resample=Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        image = (image - self.pixel_mean) / self.pixel_std
        return image

def get_dataloaders(data_dir, batch_size=4):
    processor = SAMProcessor()
    xr_dataset = SAMMedicalDataset(data_dir, modality="xray", transform=processor)
    mri_dataset = SAMMedicalDataset(data_dir, modality="mri", transform=processor)
    xr_loader = DataLoader(xr_dataset, batch_size=batch_size, shuffle=False)
    mri_loader = DataLoader(mri_dataset, batch_size=batch_size, shuffle=False)
    return xr_loader, mri_loader
