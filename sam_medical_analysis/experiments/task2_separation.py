import os
import yaml
import torch
import numpy as np
import pandas as pd
from sam_medical_analysis.models.sam_feature_extractor import SAMFeatureExtractor
from sam_medical_analysis.data.preprocess import get_dataloaders
from sam_medical_analysis.analysis.metrics import compute_silhouette, linear_probe_accuracy
from sam_medical_analysis.analysis.visualize import plot_tsne_grid

def run_task2(config):
    print("--- Running Task 2: Patch Embedding Extraction & Modality Separation ---")
    extractor = SAMFeatureExtractor(
        model_type=config['sam']['model_type'],
        checkpoint_path=config['paths']['sam_checkpoint']
    )
    layer_indices = config['analysis']['extraction_layers']
    extractor.register_hooks(layer_indices)
    xr_loader, mri_loader = get_dataloaders(config['paths']['data_dir'], batch_size=4)
    print(f"X-ray dataset size: {len(xr_loader.dataset)}")
    print(f"MRI dataset size: {len(mri_loader.dataset)}")
    all_feats = {f"block_{i}": [] for i in layer_indices}
    all_labels = []
    print("Extracting features from both modalities...")
    for loader, label in [(xr_loader, "xray"), (mri_loader, "mri")]:
        if len(loader.dataset) == 0:
            print(f"Warning: No samples found for {label}. Skipping extraction for this modality.")
            continue
        count = 0
        for i, (batch, _) in enumerate(loader):
            if i >= 12: break
            feats = extractor.extract_features(batch)
            for k, v in feats.items():
                all_feats[k].append(v.numpy())
            if label == "xray":
                all_labels.extend(["X-ray"] * batch.size(0))
            else:
                all_labels.extend(["MRI"] * batch.size(0))
            count += batch.size(0)
        print(f"Extracted {count} samples from {label}")
    metrics = []
    layer_feats_list = []
    for layer in layer_indices:
        if not all_feats[f"block_{layer}"]:
            print(f"Warning: No features found for layer {layer}. Skipping.")
            continue
        feat_block = np.concatenate(all_feats[f"block_{layer}"], axis=0)
        layer_feats_list.append(feat_block)
        sil = compute_silhouette(feat_block, all_labels)
        acc = linear_probe_accuracy(feat_block, all_labels)
        metrics.append({
            "Layer": layer,
            "Silhouette": sil,
            "LinearProbeAcc": acc
        })
        print(f"Layer {layer}: Silhouette={sil:.3f}, ProbeAcc={acc:.3f}")
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(config['paths']['metrics_dir'], "separation_metrics.csv"), index=False)
    plot_tsne_grid(
        layer_feats_list, 
        all_labels, 
        [f"Block {i}" for i in layer_indices],
        os.path.join(config['paths']['figures_dir'], "tsne_grid.png")
    )
    print("Task 2 complete. Results saved to outputs/.")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_task2(config)
