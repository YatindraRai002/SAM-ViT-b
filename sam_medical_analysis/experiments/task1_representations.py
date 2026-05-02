import os
import yaml
import torch
import pandas as pd
import numpy as np
from sam_medical_analysis.models.sam_feature_extractor import SAMFeatureExtractor
from sam_medical_analysis.data.preprocess import get_dataloaders
from sam_medical_analysis.analysis.metrics import compute_cka
from sam_medical_analysis.analysis.visualize import plot_cka_line, plot_activation_energy

def run_task1(config):
    print("--- Running Task 1: Intermediate Representation Analysis ---")
    extractor = SAMFeatureExtractor(
        model_type=config['sam']['model_type'],
        checkpoint_path=config['paths']['sam_checkpoint']
    )
    layer_indices = config['analysis']['layers']
    extractor.register_hooks(layer_indices)
    xr_loader, mri_loader = get_dataloaders(config['paths']['data_dir'], batch_size=4)
    all_xr_feats = {f"block_{i}": [] for i in layer_indices}
    all_mri_feats = {f"block_{i}": [] for i in layer_indices}
    print("Extracting X-ray features...")
    for i, (batch, _) in enumerate(xr_loader):
        feats = extractor.extract_features(batch)
        for k, v in feats.items():
            all_xr_feats[k].append(v)

    print("Extracting MRI features...")
    # No need to remove and re-register hooks if the indices are the same
    for i, (batch, _) in enumerate(mri_loader):
        feats = extractor.extract_features(batch)
        for k, v in feats.items():
            all_mri_feats[k].append(v)


    cka_results = []
    for l in layer_indices:
        f1 = torch.cat(all_xr_feats[f"block_{l}"], dim=0)
        f2 = torch.cat(all_mri_feats[f"block_{l}"], dim=0)
        cka_results.append(compute_cka(f1, f2))

    cka_df = pd.DataFrame({"Layer": layer_indices, "CKA": cka_results})
    cka_df.set_index("Layer", inplace=True)
    cka_df.to_csv(os.path.join(config['paths']['metrics_dir'], "cka_matrix.csv"))
    plot_cka_line(cka_df, os.path.join(config['paths']['figures_dir'], "cka_matrix.png"))

    energies = {'xray': [], 'mri': []}
    for l in layer_indices:
        energies['xray'].append(torch.cat(all_xr_feats[f"block_{l}"], dim=0).abs().mean().item())
        energies['mri'].append(torch.cat(all_mri_feats[f"block_{l}"], dim=0).abs().mean().item())
    plot_activation_energy(energies, layer_indices, os.path.join(config['paths']['figures_dir'], "layer_activations_plot.png"))
    print("Task 1 complete. Results saved to outputs/.")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_task1(config)
