import os
import requests
import tarfile
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

def generate_synthetic_data(data_dir, num_samples=20):
    xr_dir = Path(data_dir) / "xray"
    mri_dir = Path(data_dir) / "mri"
    os.makedirs(xr_dir, exist_ok=True)
    os.makedirs(mri_dir, exist_ok=True)
    print(f"Generating synthetic data in {data_dir}...")
    for i in range(num_samples):
        # X-ray image and mask
        xr_img = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        for j in range(10):
            xr_img[:, j*100:(j*100+20)] = 200
        from PIL import Image
        Image.fromarray(xr_img).save(xr_dir / f"xr_{i:03d}.png")
        # Generate a simple strip mask for X-ray
        xr_mask = np.zeros((1024, 1024), dtype=np.uint8)
        for j in range(10):
            xr_mask[:, j*100:(j*100+20)] = 255
        Image.fromarray(xr_mask).save(xr_dir / f"xr_{i:03d}_mask.png")

        # MRI image and mask
        mri_img = np.zeros((512, 512), dtype=np.uint8)
        xx, yy = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512))
        dist = np.sqrt(xx**2 + yy**2)
        mri_img[dist < 0.5] = 150
        mri_img[dist < 0.2] = 250
        mri_img = (mri_img + np.random.normal(0, 10, (512, 512))).clip(0, 255).astype(np.uint8)
        Image.fromarray(mri_img).save(mri_dir / f"mri_{i:03d}.png")
        # Generate a circle mask for MRI
        mri_mask = np.zeros((512, 512), dtype=np.uint8)
        mri_mask[dist < 0.5] = 255
        Image.fromarray(mri_mask).save(mri_dir / f"mri_{i:03d}_mask.png")

def download_sam_checkpoint(dest_path):
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    if os.path.exists(dest_path):
        print(f"SAM checkpoint already exists at {dest_path}")
        return
    print(f"Downloading SAM checkpoint from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc="SAM-ViT-B"
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    data_dir = config['paths']['data_dir']
    checkpoint_path = config['paths']['sam_checkpoint']
    generate_synthetic_data(data_dir)
