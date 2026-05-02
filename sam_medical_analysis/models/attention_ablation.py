import torch
import torch.nn as nn

class AttentionAblator:
    def __init__(self, model):
        self.model = model
        self.hooks = []

    def _ablate_head_hook(self, head_idx, num_heads):
        def hook(module, input, output):
            C = output.shape[-1]
            head_dim = C // num_heads
            mask = torch.ones(C, device=output.device)
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            mask[start:end] = 0
            view_shape = [1] * (len(output.shape) - 1) + [-1]
            return output * mask.view(*view_shape)
        return hook

    def apply_ablation(self, layer_idx, head_idx):
        self.remove_ablation()
        target_module = self.model.image_encoder.blocks[layer_idx].attn
        num_heads = target_module.num_heads
        hook = target_module.register_forward_hook(self._ablate_head_hook(head_idx, num_heads))
        self.hooks.append(hook)
        print(f"Ablated head {head_idx} in layer {layer_idx}.")

    def remove_ablation(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
