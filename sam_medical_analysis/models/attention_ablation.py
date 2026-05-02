import torch
import torch.nn as nn


class AttentionAblator:
    """
    Ablates individual attention heads in SAM's ViT image encoder.

    Strategy: Hook the QKV linear projection and zero out the Value (V)
    vectors for the target head. Since each head's contribution to the
    output is computed as:

        head_h_output = softmax(Q_h @ K_h^T / sqrt(d)) @ V_h

    setting V_h = 0 guarantees that head h contributes exactly zero to the
    output, with no cross-head contamination from the projection matrix.

    This is true per-head ablation — the previous approach of zeroing output
    channels AFTER the projection matrix mixed all heads together was imprecise.
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []

    def _ablate_v_hook(self, head_idx, num_heads):
        """
        Returns a forward hook for the QKV linear layer that zeroes the
        Value vectors for the specified head.

        The QKV projection output has shape (B, H, W, 3 * num_heads * head_dim).
        The layout is: [Q_h0 | Q_h1 | ... | K_h0 | K_h1 | ... | V_h0 | V_h1 | ...]
        V for head h starts at index: 2 * num_heads * head_dim + h * head_dim
        """
        def hook(module, input, output):
            total_dim = output.shape[-1]
            head_dim = total_dim // (3 * num_heads)

            # Locate V slice for the target head
            v_start = 2 * num_heads * head_dim + head_idx * head_dim
            v_end   = v_start + head_dim

            # Clone to avoid in-place modification of the computation graph
            out = output.clone()
            out[..., v_start:v_end] = 0.0
            return out

        return hook

    def apply_ablation(self, layer_idx, head_idx):
        """
        Register ablation on the QKV projection of the specified layer's
        attention module, targeting the specified head index.
        """
        self.remove_ablation()
        attn_module = self.model.image_encoder.blocks[layer_idx].attn
        num_heads = attn_module.num_heads

        hook = attn_module.qkv.register_forward_hook(
            self._ablate_v_hook(head_idx, num_heads)
        )
        self.hooks.append(hook)
        print(f"Ablated head {head_idx} in layer {layer_idx} "
              f"(V-zeroing, head_dim={attn_module.qkv.out_features // (3 * num_heads)}).")

    def remove_ablation(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
