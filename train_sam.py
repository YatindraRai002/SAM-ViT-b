import os
import yaml
import torch
from torch.utils.data import DataLoader
from sam_medical_analysis.models.sam_feature_extractor import SAMFeatureExtractor
from sam_medical_analysis.models.sam_trainer import SAMTrainer
from sam_medical_analysis.data.preprocess import get_dataloaders, SAMMedicalDataset, SAMProcessor
from segment_anything import sam_model_registry

def train_sam(config):
    print("--- Starting SAM Medical Fine-Tuning ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model
    print(f"Loading model {config['sam']['model_type']}...")
    sam = sam_model_registry[config['sam']['model_type']](checkpoint=config['paths']['sam_checkpoint'])
    sam.to(device)

    # 2. Prepare Data
    # We create a separate training set by splitting the synthetic/real data
    # Since get_dataloaders is for analysis, we create custom training loaders
    processor = SAMProcessor()

    # Create Training and Validation sets
    # Split logic: 80% train, 20% val
    train_xr = SAMMedicalDataset(config['paths']['data_dir'], modality="xray", transform=processor)
    train_mri = SAMMedicalDataset(config['paths']['data_dir'], modality="mri", transform=processor)

    # Combine modalities for a general medical trainer
    train_images = []
    train_masks = []

    # Simple combine logic for demonstration
    # Use all available samples per modality
    n_samples = min(len(train_xr), 20)
    print(f"Loading {n_samples} samples per modality...", flush=True)
    for i in range(n_samples):
        img, mask = train_xr[i]
        train_images.append(img)
        train_masks.append(mask)

    for i in range(n_samples):
        img, mask = train_mri[i]
        train_images.append(img)
        train_masks.append(mask)

    # Create a simple DataLoader wrapper
    class SimpleTrainSet:
        def __init__(self, imgs, msks):
            self.imgs = imgs
            self.msks = msks
        def __len__(self): return len(self.imgs)
        def __getitem__(self, idx): return self.imgs[idx], self.msks[idx]

    # batch_size=1 to stay within RAM on CPU-only machines
    train_set = SimpleTrainSet(train_images, train_masks)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    # For validation, we use a small subset
    val_set = SimpleTrainSet(train_images[:2], train_masks[:2])
    val_loader = DataLoader(val_set, batch_size=1)

    # 3. Initialize Trainer
    trainer = SAMTrainer(sam, config)

    # 4. Start Training
    epochs = 5  # Full training run
    trainer.fit(train_loader, val_loader, epochs=epochs)

    print(f"Training complete. Best model saved to {config['paths']['output_dir']}/best_sam_medical.pth")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_sam(config)
