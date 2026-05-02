import os
import sys
# Ensure project root (d:/li) is on the path when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry
from sam_medical_analysis.data.preprocess import SAMMedicalDataset, SAMProcessor
from sam_medical_analysis.analysis.metrics import compute_dice


# --------------------------------------------------------------------------- #
#  Prompt generation helpers                                                    #
# --------------------------------------------------------------------------- #

def _no_prompt(mask_tensor, device):
    """No guidance — zero-vector sparse and dense embeddings."""
    sparse = torch.zeros((1, 1, 256), device=device)
    dense  = torch.zeros((1, 256, 64, 64), device=device)
    return sparse, dense


def _center_point_prompt(mask_tensor, sam, device):
    """
    Single point at the centroid of the GT mask foreground region.
    If the mask is empty, falls back to the image centre.
    """
    binary = (mask_tensor > 0.5)
    if binary.any():
        ys, xs = torch.where(binary)
        cy = ys.float().mean().long().item()
        cx = xs.float().mean().long().item()
    else:
        cy, cx = mask_tensor.shape[0] // 2, mask_tensor.shape[1] // 2

    # SAM prompt encoder expects (B, N, 2) coords and (B, N) labels
    coords  = torch.tensor([[[cx, cy]]], dtype=torch.float, device=device)  # (1, 1, 2)
    labels  = torch.tensor([[1]], dtype=torch.int, device=device)            # (1, 1)

    with torch.no_grad():
        sparse, dense = sam.prompt_encoder(
            points=(coords, labels),
            boxes=None,
            masks=None,
        )
    return sparse, dense


def _gt_box_prompt(mask_tensor, sam, device):
    """Tight bounding box derived from the GT mask foreground region."""
    binary = (mask_tensor > 0.5)
    if not binary.any():
        return _no_prompt(mask_tensor, device)

    rows = binary.any(dim=1)
    cols = binary.any(dim=0)
    y1 = torch.where(rows)[0][0].item()
    y2 = torch.where(rows)[0][-1].item()
    x1 = torch.where(cols)[0][0].item()
    x2 = torch.where(cols)[0][-1].item()

    box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float, device=device)  # (1, 4)

    with torch.no_grad():
        sparse, dense = sam.prompt_encoder(
            points=None,
            boxes=box,
            masks=None,
        )
    return sparse, dense


def _noisy_box_prompt(mask_tensor, sam, device, jitter=0.10):
    """
    GT bounding box with ±jitter% random noise on each coordinate.
    Simulates realistic prompts where the user draws an approximate box.
    Coordinates are clamped to image bounds after jittering.
    """
    binary = (mask_tensor > 0.5)
    if not binary.any():
        return _no_prompt(mask_tensor, device)

    rows = binary.any(dim=1)
    cols = binary.any(dim=0)
    y1 = float(torch.where(rows)[0][0].item())
    y2 = float(torch.where(rows)[0][-1].item())
    x1 = float(torch.where(cols)[0][0].item())
    x2 = float(torch.where(cols)[0][-1].item())

    w = x2 - x1
    h = y2 - y1
    H, W = mask_tensor.shape

    # Add uniform noise proportional to box size
    x1 = float(np.clip(x1 + np.random.uniform(-jitter * w, jitter * w), 0, W - 1))
    y1 = float(np.clip(y1 + np.random.uniform(-jitter * h, jitter * h), 0, H - 1))
    x2 = float(np.clip(x2 + np.random.uniform(-jitter * w, jitter * w), 0, W - 1))
    y2 = float(np.clip(y2 + np.random.uniform(-jitter * h, jitter * h), 0, H - 1))

    box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float, device=device)

    with torch.no_grad():
        sparse, dense = sam.prompt_encoder(
            points=None,
            boxes=box,
            masks=None,
        )
    return sparse, dense


# --------------------------------------------------------------------------- #
#  Inference helper                                                             #
# --------------------------------------------------------------------------- #

def _run_inference(sam, image_embedding, sparse, dense, device):
    """
    Run the mask decoder for one image embedding + prompt pair.
    Returns a binary mask tensor of shape (H_orig, W_orig) at 1024x1024.
    """
    image_pe = sam.prompt_encoder.get_dense_pe()

    with torch.no_grad():
        low_res, _ = sam.mask_decoder(
            image_embeddings=image_embedding,    # (1, 256, 64, 64)
            image_pe=image_pe,                   # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )

    pred = torch.nn.functional.interpolate(
        low_res, size=(1024, 1024), mode="bilinear", align_corners=False
    )
    return (torch.sigmoid(pred[0, 0]) > 0.5).float()  # (1024, 1024) binary


# --------------------------------------------------------------------------- #
#  Main task                                                                    #
# --------------------------------------------------------------------------- #

def run_task6(config):
    print("--- Running Task 6: Prompt Sensitivity Analysis ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load fine-tuned model (or base SAM if not available)
    sam = sam_model_registry[config['sam']['model_type']](
        checkpoint=config['paths']['sam_checkpoint']
    )
    finetuned_path = os.path.join(config['paths']['output_dir'], "best_sam_medical.pth")
    if os.path.exists(finetuned_path):
        sam.load_state_dict(torch.load(finetuned_path, map_location="cpu"))
        print("Loaded fine-tuned weights.", flush=True)
    else:
        print("Fine-tuned weights not found — using base SAM.", flush=True)

    sam.to(device)
    sam.eval()

    # Freeze all parameters — inference only
    for p in sam.parameters():
        p.requires_grad = False

    processor = SAMProcessor()

    # Test on all available samples from both modalities
    strategies = {
        "No Prompt":       lambda m: _no_prompt(m, device),
        "Center Point":    lambda m: _center_point_prompt(m, sam, device),
        "GT Box":          lambda m: _gt_box_prompt(m, sam, device),
        "Noisy Box (±10%)": lambda m: _noisy_box_prompt(m, sam, device),
    }

    modalities = [
        ("X-ray", "xray"),
        ("MRI",   "mri"),
    ]

    results = []
    np.random.seed(42)  # Reproducible noise for noisy box

    for mod_label, mod_key in modalities:
        dataset = SAMMedicalDataset(
            config['paths']['data_dir'],
            modality=mod_key,
            transform=processor
        )
        print(f"\nEvaluating {mod_label} ({len(dataset)} samples)...", flush=True)

        for idx in range(len(dataset)):
            image_tensor, mask_tensor = dataset[idx]
            image_tensor = image_tensor.unsqueeze(0).to(device)   # (1, 3, 1024, 1024)
            mask_tensor  = mask_tensor.to(device)                  # (1024, 1024)

            # Run image encoder ONCE per image — reuse embedding across all strategies
            with torch.no_grad():
                image_embedding = sam.image_encoder(image_tensor)  # (1, 256, 64, 64)

            for strategy_name, prompt_fn in strategies.items():
                sparse, dense = prompt_fn(mask_tensor)
                pred_mask = _run_inference(sam, image_embedding, sparse, dense, device)
                dice = compute_dice(pred_mask, mask_tensor).item()

                results.append({
                    "Modality": mod_label,
                    "Strategy": strategy_name,
                    "Sample":   idx,
                    "Dice":     dice,
                })

            if (idx + 1) % 5 == 0:
                print(f"  {mod_label}: {idx + 1}/{len(dataset)} samples done", flush=True)

    # ------------------------------------------------------------------ #
    # Save results                                                         #
    # ------------------------------------------------------------------ #
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(config['paths']['metrics_dir'], "prompt_sensitivity.csv"),
        index=False
    )

    # Summary: mean Dice per modality × strategy
    summary = results_df.groupby(["Modality", "Strategy"])["Dice"].agg(["mean", "std"]).reset_index()
    summary.columns = ["Modality", "Strategy", "Mean Dice", "Std Dice"]
    print("\nPrompt Sensitivity Results:")
    print(summary.to_string(index=False))

    summary.to_csv(
        os.path.join(config['paths']['metrics_dir'], "prompt_sensitivity_summary.csv"),
        index=False
    )

    # ------------------------------------------------------------------ #
    # Plot: grouped bar chart                                              #
    # ------------------------------------------------------------------ #
    strategy_order  = list(strategies.keys())
    modality_labels = [m[0] for m in modalities]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x       = np.arange(len(strategy_order))
    width   = 0.35
    n_mods  = len(modality_labels)
    offsets = np.linspace(-(n_mods - 1) * width / 2, (n_mods - 1) * width / 2, n_mods)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (mod_label, _) in enumerate(modalities):
        mod_data = summary[summary["Modality"] == mod_label].set_index("Strategy")
        means = [mod_data.loc[s, "Mean Dice"] if s in mod_data.index else 0
                 for s in strategy_order]
        stds  = [mod_data.loc[s, "Std Dice"]  if s in mod_data.index else 0
                 for s in strategy_order]
        bars = ax.bar(
            x + offsets[i], means, width,
            label=mod_label,
            color=colors[i % len(colors)],
            yerr=stds, capsize=4,
            alpha=0.85, edgecolor="white", linewidth=0.8,
        )
        # Annotate bar tops with mean value
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Prompt Strategy", fontsize=12)
    ax.set_ylabel("Mean Dice Score", fontsize=12)
    ax.set_title("Segmentation Quality by Prompt Strategy\n(Fine-tuned SAM ViT-B)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_order, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(config['paths']['figures_dir'], "prompt_sensitivity.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {out_path}", flush=True)
    print("Task 6 complete. Results saved to outputs/.")


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    run_task6(config)
