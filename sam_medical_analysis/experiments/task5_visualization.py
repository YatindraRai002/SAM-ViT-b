import os
import sys
# Ensure project root (d:/li) is on the path when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from segment_anything import sam_model_registry
from sam_medical_analysis.data.preprocess import SAMProcessor


def _load_model(config, device):
    """
    Load SAM. Uses fine-tuned weights if available, otherwise base model.
    """
    sam = sam_model_registry[config['sam']['model_type']](
        checkpoint=config['paths']['sam_checkpoint']
    )
    finetuned_path = os.path.join(config['paths']['output_dir'], "best_sam_medical.pth")
    if os.path.exists(finetuned_path):
        sam.load_state_dict(torch.load(finetuned_path, map_location="cpu"))
        print("Loaded fine-tuned weights for visualization.")
    else:
        print("Fine-tuned weights not found — using base SAM.")
    sam.to(device)
    sam.eval()
    return sam


def _extract_spatial_features(sam, image_tensor, layer_indices, device):
    """
    Extract spatial feature maps (H, W, C) from the specified transformer
    blocks using forward hooks. Unlike Phase 2, spatial structure is preserved
    (no global average pooling) so we can generate spatial heatmaps.
    """
    spatial_feats = {}
    hooks = []

    def make_hook(layer_id):
        def hook(module, input, output):
            # SAM ViT block output: (B, H, W, C)
            spatial_feats[layer_id] = output.detach().cpu()
        return hook

    for idx in layer_indices:
        h = sam.image_encoder.blocks[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    with torch.no_grad():
        sam.image_encoder(image_tensor)

    for h in hooks:
        h.remove()

    return spatial_feats


def _make_overlay(orig_np, importance_2d, alpha=0.55, colormap="hot"):
    """
    Blend the spatial importance map over the original image as a coloured
    heatmap. Returns an (H, W, 3) float array in [0, 1].

    alpha controls the heatmap opacity (0 = image only, 1 = heatmap only).
    """
    # Normalise importance to [0, 1]
    imp = importance_2d.astype(np.float32)
    imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)

    # Resize to match original image resolution
    h, w = orig_np.shape[:2]
    imp_resized = np.array(
        Image.fromarray((imp * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0

    heatmap = plt.colormaps[colormap](imp_resized)[:, :, :3]  # (H, W, 3) RGB

    base = orig_np[:, :, :3] / 255.0
    overlay = (1 - alpha) * base + alpha * heatmap
    return np.clip(overlay, 0, 1)


def run_task5(config):
    print("--- Running Task 5: Feature Activation Map Visualization ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = _load_model(config, device)
    processor = SAMProcessor()

    layers_to_visualize = [0, 4, 8, 11]

    test_images = [
        ("X-ray", os.path.join(config['paths']['data_dir'], "xray", "xr_000.png"),
                  os.path.join(config['paths']['data_dir'], "xray", "xr_000_mask.png")),
        ("MRI",   os.path.join(config['paths']['data_dir'], "mri",  "mri_000.png"),
                  os.path.join(config['paths']['data_dir'], "mri",  "mri_000_mask.png")),
    ]

    for modality, img_path, mask_path in test_images:
        print(f"  Generating activation maps for {modality}...", flush=True)

        orig_img = Image.open(img_path).convert("RGB")
        orig_np  = np.array(orig_img)

        tensor = processor(orig_img).unsqueeze(0).to(device)

        spatial_feats = _extract_spatial_features(sam, tensor, layers_to_visualize, device)

        # Load GT mask for reference
        mask_np = np.array(Image.open(mask_path).convert("L"))

        n_cols = len(layers_to_visualize) + 2  # original + layers + GT mask
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))

        # Column 0: original image
        axes[0].imshow(orig_np, cmap="gray" if orig_np.mean() < 100 else None)
        axes[0].set_title("Original", fontsize=10)
        axes[0].axis("off")

        # Columns 1..N: one per layer
        for i, layer_idx in enumerate(layers_to_visualize):
            feat = spatial_feats[layer_idx][0]     # (H_feat, W_feat, C)

            # Mean absolute activation across channels → spatial importance
            importance = feat.abs().mean(dim=-1).numpy()   # (H_feat, W_feat)

            overlay = _make_overlay(orig_np, importance)
            axes[i + 1].imshow(overlay)
            axes[i + 1].set_title(f"Layer {layer_idx}", fontsize=10)
            axes[i + 1].axis("off")

        # Last column: GT mask
        axes[-1].imshow(mask_np, cmap="gray")
        axes[-1].set_title("GT Mask", fontsize=10)
        axes[-1].axis("off")

        fig.suptitle(f"Feature Activation Maps — {modality}", fontsize=13, y=1.01)
        plt.tight_layout()

        out_fname = f"activation_maps_{modality.lower().replace('-', '').replace(' ', '')}.png"
        out_path  = os.path.join(config['paths']['figures_dir'], out_fname)
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}", flush=True)

    print("Task 5 complete. Activation maps saved to outputs/figures/.")


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    run_task5(config)
