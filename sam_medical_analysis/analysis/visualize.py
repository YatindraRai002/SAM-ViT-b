import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

def plot_cka_line(cka_df, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(cka_df.index, cka_df.iloc[:, 0], marker='o', linewidth=2, color='blue')
    plt.title("CKA Similarity: X-ray vs MRI (Same-Layer Comparison)")
    plt.xlabel("Layer Index")
    plt.ylabel("CKA Similarity")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def plot_tsne_grid(features, labels, layer_names, output_path):
    n_layers = len(features)
    cols = 2
    rows = (n_layers + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten()
    
    for i, (feat, name) in enumerate(zip(features, layer_names)):
        tsne = TSNE(n_components=2, perplexity=min(30, len(feat)-1), random_state=42)
        embed = tsne.fit_transform(feat)
        
        df = pd.DataFrame(embed, columns=['dim1', 'dim2'])
        df['modality'] = labels
        
        sns.scatterplot(data=df, x='dim1', y='dim2', hue='modality', ax=axes[i])
        axes[i].set_title(f"t-SNE: {name}")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_activation_energy(energies, layer_indices, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(layer_indices, energies['xray'], label='X-ray', marker='o')
    plt.plot(layer_indices, energies['mri'], label='MRI', marker='s')
    plt.title("Activation Energy Across Transformer Blocks")
    plt.xlabel("Block Index")
    plt.ylabel("Mean Activation Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_ablation_heatmap(results, output_path):
    plt.figure(figsize=(12, 8))
    sns.heatmap(results, annot=True, cmap="RdYlGn_r", fmt=".2f")
    plt.title("Ablation Impact: Dice Score Drop (%)")
    plt.xlabel("Modality")
    plt.ylabel("Head Index")
    plt.savefig(output_path)
    plt.close()
