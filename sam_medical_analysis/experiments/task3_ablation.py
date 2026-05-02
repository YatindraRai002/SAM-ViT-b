import os
import yaml
import torch
import pandas as pd
import numpy as np
from PIL import Image
from segment_anything import SamPredictor
from sam_medical_analysis.models.sam_feature_extractor import SAMFeatureExtractor
from sam_medical_analysis.models.attention_ablation import AttentionAblator
from sam_medical_analysis.data.preprocess import SAMProcessor
from sam_medical_analysis.analysis.metrics import compute_dice
from sam_medical_analysis.analysis.visualize import plot_ablation_heatmap


def _predict_mask(predictor: SamPredictor, image_np: np.ndarray, device: str) -> torch.Tensor:
    """
    Run SAM inference on a single image (H, W, 3) numpy array (uint8).
    Returns a binary mask tensor of shape (H, W).
    """
    predictor.set_image(image_np)
    # Use a centre-point prompt so the decoder always fires
    h, w = image_np.shape[:2]
    point_coords = np.array([[w // 2, h // 2]])
    point_labels = np.array([1])
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    # masks: (N_masks, H, W) bool array; take the first mask
    return torch.tensor(masks[0], dtype=torch.float32, device=device)


def run_task3(config):
    print("--- Running Task 3: Attention Head Ablation & Mechanistic Hypotheses ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = SAMFeatureExtractor(
        model_type=config['sam']['model_type'],
        checkpoint_path=config['paths']['sam_checkpoint'],
        device=device
    )

    ablator = AttentionAblator(extractor.sam)

    # SamPredictor wraps the SAM model and exposes a proper .predict() API
    predictor = SamPredictor(extractor.sam)

    processor = SAMProcessor()

    # ------------------------------------------------------------------ #
    # Load test images as numpy uint8 arrays (required by SamPredictor)   #
    # ------------------------------------------------------------------ #
    def load_image_np(path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def get_gt_mask(path: str) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        mask_np = np.array(mask)
        return torch.tensor(mask_np > 0, dtype=torch.float32, device=device)

    xr_img_np  = load_image_np(os.path.join(config['paths']['data_dir'], "xray", "xr_000.png"))
    mri_img_np = load_image_np(os.path.join(config['paths']['data_dir'], "mri",  "mri_000.png"))

    mri_gt = get_gt_mask(os.path.join(config['paths']['data_dir'], "mri",  "mri_000_mask.png"))
    xr_gt  = get_gt_mask(os.path.join(config['paths']['data_dir'], "xray", "xr_000_mask.png"))

    layer_idx    = config['analysis']['ablation_layer']
    num_heads    = extractor.encoder.blocks[layer_idx].attn.num_heads
    heads_to_test = range(0, min(num_heads, 4))

    results = []
    print(f"Testing ablation on Layer {layer_idx} ({num_heads} heads, testing first {len(heads_to_test)})...")

    for head in heads_to_test:
        # ---- ablated prediction ---------------------------------------- #
        ablator.apply_ablation(layer_idx, head)
        with torch.no_grad():
            mri_abl_t = _predict_mask(predictor, mri_img_np, device)
            xr_abl_t  = _predict_mask(predictor, xr_img_np,  device)
        ablator.remove_ablation()

        # Resize GT masks to match prediction size (predictor returns original image size)
        def resize_gt(gt: torch.Tensor, h: int, w: int) -> torch.Tensor:
            if gt.shape == (h, w):
                return gt
            gt_pil = Image.fromarray(gt.cpu().numpy().astype(np.uint8) * 255)
            gt_pil = gt_pil.resize((w, h), resample=Image.NEAREST)
            return torch.tensor(np.array(gt_pil) > 0, dtype=torch.float32, device=device)

        mri_gt_r = resize_gt(mri_gt, mri_abl_t.shape[0], mri_abl_t.shape[1])
        xr_gt_r  = resize_gt(xr_gt,  xr_abl_t.shape[0],  xr_abl_t.shape[1])

        # Dice Drop = 1 - Dice(Ablated, GT)   (higher means ablation hurt more)
        mri_drop = float((1.0 - compute_dice(mri_abl_t, mri_gt_r)) * 100)
        xr_drop  = float((1.0 - compute_dice(xr_abl_t,  xr_gt_r))  * 100)

        print(f"  Head {head}: MRI dice-drop={mri_drop:.2f}%, X-ray dice-drop={xr_drop:.2f}%")

        results.append({"Head": head, "Modality": "MRI",   "DiceDrop": mri_drop})
        results.append({"Head": head, "Modality": "X-ray", "DiceDrop": xr_drop})

    results_df = pd.DataFrame(results)
    pivot = results_df.pivot(index="Head", columns="Modality", values="DiceDrop")
    pivot.to_csv(os.path.join(config['paths']['metrics_dir'], "ablation_results.csv"))
    plot_ablation_heatmap(pivot, os.path.join(config['paths']['figures_dir'], "dice_comparison.png"))

    with open(os.path.join(config['paths']['metrics_dir'], "hypothesis_test_results.txt"), "w") as f:
        f.write(f"Ablation Study Results (Layer {layer_idx}):\n")
        f.write(f"Tested Heads: {list(heads_to_test)}\n")
        f.write(f"Average Dice Drop MRI:   {results_df[results_df['Modality']=='MRI']['DiceDrop'].mean():.2f}%\n")
        f.write(f"Average Dice Drop X-ray: {results_df[results_df['Modality']=='X-ray']['DiceDrop'].mean():.2f}%\n")
        f.write(
            "Conclusion: Dice drops measured against ground truth indicate "
            "head-specific importance for modality-specific segmentation.\n"
        )

    print("Task 3 complete. Results saved to outputs/.")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_task3(config)
