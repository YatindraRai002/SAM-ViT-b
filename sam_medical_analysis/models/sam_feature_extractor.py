import torch
from segment_anything import sam_model_registry
import os

class SAMFeatureExtractor:
    def __init__(self, model_type="vit_h", checkpoint_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)
        self.sam.eval()
        self.encoder = self.sam.image_encoder
        self.features = {}
        self.hooks = []

    def _get_hook(self, name):
        def hook(module, input, output):
            # output is (B, H, W, C) or (B, N, C) for ViT blocks
            # Global Average Pooling over tokens
            B = output.shape[0]
            C = output.shape[-1]
            pooled = output.view(B, -1, C).mean(dim=1)
            self.features[name] = pooled.detach().cpu()
        return hook

    def register_hooks(self, layer_indices):
        self.remove_hooks()
        for idx in layer_indices:
            if idx < len(self.encoder.blocks):
                hook = self.encoder.blocks[idx].register_forward_hook(self._get_hook(f"block_{idx}"))
                self.hooks.append(hook)
            else:
                print(f"Warning: Layer index {idx} out of range.")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}

    def extract_features(self, batch):
        batch = batch.to(self.device)
        self.features = {}
        with torch.no_grad():
            self.encoder(batch)
        return self.features


if __name__ == "__main__":
    extractor = SAMFeatureExtractor(checkpoint_path=None)
    print(f"Encoder has {len(extractor.encoder.blocks)} blocks.")
