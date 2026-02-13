"""
Drop-in replacement for timm's Attention that exposes intermediate values.

Usage:
    from src.utils.hookable_attention import replace_attention_modules
    replace_attention_modules(model.visual)  # mutates in-place, same weights
"""

from typing import Optional

import torch
from timm.layers import Attention as TimmAttention


class HookableAttention(TimmAttention):
    """Extends timm Attention to store intermediates during forward.

    After each forward pass the following attributes are populated:
        _q:            (B, num_heads, N, head_dim)  — queries after norm
        _k:            (B, num_heads, N, head_dim)  — keys after norm
        _v:            (B, num_heads, N, head_dim)  — values
        _attn_weights: (B, num_heads, N, N)         — post-softmax attention
        _attn_output:  (B, N, C)                    — after head merge, before proj
    """

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        self._q = q
        self._k = k
        self._v = v

        # Always unfused so we can capture attention weights
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        self._attn_weights = attn

        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        self._attn_output = x

        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def clear_intermediates(self):
        self._q = None
        self._k = None
        self._v = None
        self._attn_weights = None
        self._attn_output = None


def replace_attention_modules(visual_encoder):
    """Replace all timm Attention modules in a ViT with HookableAttention (in-place).

    Swaps the __class__ of each Attention instance so the existing weights,
    submodules, and parent references stay intact — no copying needed.

    Returns:
        List of (name, HookableAttention) for convenience.
    """
    replaced = []
    for name, module in visual_encoder.named_modules():
        if type(module) is TimmAttention:
            module.__class__ = HookableAttention
            replaced.append((name, module))

    print(f'Replaced {len(replaced)} Attention modules with HookableAttention')
    return replaced


if __name__ == '__main__':
    from open_clip import create_model_from_pretrained

    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    visual = model.visual.eval().requires_grad_(False)

    replaced = replace_attention_modules(visual)

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = visual(x)

    print(f'Output shape: {out.shape}')
    name, h = replaced[0]
    print(f'{name}: q={h._q.shape}, attn_weights={h._attn_weights.shape}')

    # Numerical equivalence against original
    model2, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    with torch.no_grad():
        out2 = model2.visual.eval()(x)
    print(f'Max diff vs original: {(out - out2).abs().max().item():.2e}')
