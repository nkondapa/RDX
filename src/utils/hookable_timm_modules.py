"""
Drop-in replacements for timm's Attention and Mlp that expose intermediate values.

Usage:
    from src.utils.hookable_timm_modules import replace_attention_modules, replace_mlp_modules
    replace_attention_modules(model.visual)
    replace_mlp_modules(model.visual)
"""

from typing import Optional

import torch
from timm.layers import Attention as TimmAttention
from timm.layers import Mlp as TimmMlp


class HookableAttention(TimmAttention):
    """Extends timm Attention to store intermediates during forward.

    After each forward pass the following attributes are populated:
        _q:            (B, num_heads, N, head_dim)  — queries after norm
        _k:            (B, num_heads, N, head_dim)  — keys after norm
        _v:            (B, num_heads, N, head_dim)  — values
        _attn_weights: (B, num_heads, N, N)         — post-softmax attention
        _attn_output:  (B, N, C)                    — after head merge, before proj

    Hook functions can be registered per intermediate via register_hook().
    Each hook receives the tensor and must return a (possibly modified) tensor.
    Hooks are called after the value is stored, so _q/_k/etc. hold the
    *pre-hook* value and the *post-hook* value is what flows forward.

    Example:
        def zero_head_3(attn_weights):
            attn_weights[:, 3, :, :] = 0
            return attn_weights

        attn_module.register_hook('attn_weights', zero_head_3)
    """

    # Valid hook points
    HOOK_POINTS = ('q', 'k', 'v', 'attn_weights', 'attn_output')

    def _ensure_state(self):
        if not hasattr(self, '_hooks'):
            self._hooks = {k: [] for k in self.HOOK_POINTS}
        if not hasattr(self, '_save_intermediates'):
            self._save_intermediates = False

    @property
    def save_intermediates(self):
        self._ensure_state()
        return self._save_intermediates

    @save_intermediates.setter
    def save_intermediates(self, value):
        self._ensure_state()
        self._save_intermediates = value

    def register_hook(self, name, fn):
        """Register a hook on an intermediate value.

        Args:
            name: One of 'q', 'k', 'v', 'attn_weights', 'attn_output'.
            fn: Callable(tensor) -> tensor. Receives and must return the
                intermediate value. May modify it in-place and return it.
        """
        assert name in self.HOOK_POINTS, f'{name} not in {self.HOOK_POINTS}'
        self._ensure_state()
        self._hooks[name].append(fn)

    def clear_hooks(self, name=None):
        """Remove hooks. If name is None, clears all hook points."""
        self._ensure_state()
        if name is None:
            self._hooks = {k: [] for k in self.HOOK_POINTS}
        else:
            self._hooks[name] = []

    def _apply_hooks(self, name, tensor):
        self._ensure_state()
        for fn in self._hooks[name]:
            tensor = fn(tensor)
        return tensor

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._ensure_state()
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self._save_intermediates:
            self._q = q
            self._k = k
            self._v = v
        q = self._apply_hooks('q', q)
        k = self._apply_hooks('k', k)
        v = self._apply_hooks('v', v)

        # Always unfused so we can capture attention weights
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        if self._save_intermediates:
            self._attn_weights = attn
        attn = self._apply_hooks('attn_weights', attn)

        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        if self._save_intermediates:
            self._attn_output = x
        x = self._apply_hooks('attn_output', x)

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


class HookableMlp(TimmMlp):
    """Extends timm Mlp to store intermediates during forward.

    After each forward pass the following attributes are populated:
        _fc1_out:  (B, N, hidden_features)  — after first linear
        _act_out:  (B, N, hidden_features)  — after activation
        _fc2_out:  (B, N, out_features)     — after second linear (before drop2)

    Hook functions work the same as HookableAttention.register_hook().

    Example:
        def clamp_hidden(fc1_out):
            return fc1_out.clamp(-1, 1)

        mlp_module.register_hook('fc1_out', clamp_hidden)
    """

    HOOK_POINTS = ('fc1_out', 'act_out', 'fc2_out')

    def _ensure_state(self):
        if not hasattr(self, '_hooks'):
            self._hooks = {k: [] for k in self.HOOK_POINTS}
        if not hasattr(self, '_save_intermediates'):
            self._save_intermediates = False

    @property
    def save_intermediates(self):
        self._ensure_state()
        return self._save_intermediates

    @save_intermediates.setter
    def save_intermediates(self, value):
        self._ensure_state()
        self._save_intermediates = value

    def register_hook(self, name, fn):
        """Register a hook on an intermediate value.

        Args:
            name: One of 'fc1_out', 'act_out', 'fc2_out'.
            fn: Callable(tensor) -> tensor.
        """
        assert name in self.HOOK_POINTS, f'{name} not in {self.HOOK_POINTS}'
        self._ensure_state()
        self._hooks[name].append(fn)

    def clear_hooks(self, name=None):
        """Remove hooks. If name is None, clears all hook points."""
        self._ensure_state()
        if name is None:
            self._hooks = {k: [] for k in self.HOOK_POINTS}
        else:
            self._hooks[name] = []

    def _apply_hooks(self, name, tensor):
        self._ensure_state()
        for fn in self._hooks[name]:
            tensor = fn(tensor)
        return tensor

    def forward(self, x):
        self._ensure_state()

        x = self.fc1(x)
        if self._save_intermediates:
            self._fc1_out = x
        x = self._apply_hooks('fc1_out', x)

        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        if self._save_intermediates:
            self._act_out = x
        x = self._apply_hooks('act_out', x)

        x = self.fc2(x)
        if self._save_intermediates:
            self._fc2_out = x
        x = self._apply_hooks('fc2_out', x)

        x = self.drop2(x)
        return x

    def clear_intermediates(self):
        self._fc1_out = None
        self._act_out = None
        self._fc2_out = None


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


def replace_mlp_modules(visual_encoder):
    """Replace all timm Mlp modules in a ViT with HookableMlp (in-place).

    Swaps the __class__ of each Mlp instance so the existing weights,
    submodules, and parent references stay intact — no copying needed.

    Returns:
        List of (name, HookableMlp) for convenience.
    """
    replaced = []
    for name, module in visual_encoder.named_modules():
        if type(module) is TimmMlp:
            module.__class__ = HookableMlp
            replaced.append((name, module))

    print(f'Replaced {len(replaced)} Mlp modules with HookableMlp')
    return replaced


if __name__ == '__main__':
    from open_clip import create_model_from_pretrained

    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    visual = model.visual.eval().requires_grad_(False)

    replaced_attn = replace_attention_modules(visual)
    replaced_mlp = replace_mlp_modules(visual)

    x = torch.randn(2, 3, 224, 224)

    # Verify intermediates are NOT saved by default
    with torch.no_grad():
        out = visual(x)
    print(f'\nOutput shape: {out.shape}')
    assert not hasattr(replaced_attn[0][1], '_q') or replaced_attn[0][1]._q is None, \
        'Should not save intermediates by default'
    print('Confirmed: intermediates not saved by default.')

    # Enable saving and verify intermediates are captured
    for _, h in replaced_attn:
        h.save_intermediates = True
    for _, m in replaced_mlp:
        m.save_intermediates = True

    with torch.no_grad():
        out = visual(x)

    name, h = replaced_attn[0]
    print(f'{name}: q={h._q.shape}, attn_weights={h._attn_weights.shape}')

    name_m, m = replaced_mlp[0]
    print(f'{name_m}: fc1_out={m._fc1_out.shape}, act_out={m._act_out.shape}, fc2_out={m._fc2_out.shape}')

    # Disable saving again for remaining tests
    for _, h in replaced_attn:
        h.save_intermediates = False
    for _, m in replaced_mlp:
        m.save_intermediates = False

    # Numerical equivalence against original (no hooks)
    model2, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    with torch.no_grad():
        out2 = model2.visual.eval()(x)
    print(f'\nMax diff vs original (no hooks): {(out - out2).abs().max().item():.2e}')

    # Verify attention hooks modify the forward pass
    def zero_head_0(attn_weights):
        attn_weights[:, 0, :, :] = 0
        return attn_weights

    replaced_attn[0][1].register_hook('attn_weights', zero_head_0)
    with torch.no_grad():
        out_hooked = visual(x)
    diff = (out - out_hooked).abs().max().item()
    print(f'Max diff with attn hook (zero head 0 in block 0): {diff:.2e}')
    assert diff > 1e-3, 'Attn hook should change the output'
    replaced_attn[0][1].clear_hooks()
    print('Attention hook intervention verified.')

    # Verify MLP hooks modify the forward pass
    def clamp_hidden(fc1_out):
        return fc1_out.clamp(-1, 1)

    replaced_mlp[0][1].register_hook('fc1_out', clamp_hidden)
    with torch.no_grad():
        out_mlp_hooked = visual(x)
    diff_mlp = (out - out_mlp_hooked).abs().max().item()
    print(f'Max diff with mlp hook (clamp fc1 in block 0): {diff_mlp:.2e}')
    assert diff_mlp > 1e-3, 'MLP hook should change the output'
    replaced_mlp[0][1].clear_hooks()
    print('MLP hook intervention verified.')

    # Clean up and verify back to normal
    with torch.no_grad():
        out_clean = visual(x)
    print(f'\nMax diff after clear_hooks: {(out - out_clean).abs().max().item():.2e}')
