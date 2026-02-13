"""
Shared utilities for RSNA BiomedCLIP experiments.

Provides model/dataset loading, transformer block discovery, and activation caching.
"""

import argparse
import os

import torch
from open_clip import create_model_from_pretrained
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.rsna_pneumonia import RSNAPneumoniaDataset


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args(description=None):
    """Shared CLI arguments common to all RSNA experiments."""
    p = argparse.ArgumentParser(description=description or __doc__)
    p.add_argument('--data_root', type=str,
                   default='/media/nkondapa/SSD2/data/RSNA',
                   help='Path to RSNA dataset root')
    p.add_argument('--output_dir', type=str, default=None,
                   help='Where to save cached activations and results')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p


# ── Model / Dataset Loading ─────────────────────────────────────────────────
def load_model(device='cuda'):
    """Load BiomedCLIP and return (visual_encoder, preprocess)."""
    print('Loading BiomedCLIP ...')
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    visual = model.visual.to(device).eval().requires_grad_(False)
    return visual, preprocess


def load_dataset(root, preprocess, split='train', binary=True):
    """Load RSNA Pneumonia dataset with the given transform."""
    return RSNAPneumoniaDataset(
        root=root, split=split, transform=preprocess, binary=binary
    )


# ── Helpers ──────────────────────────────────────────────────────────────────
def cls_token(x):
    """Post-activation: extract CLS token (index 0) from ViT block output."""
    return x[:, 0, :].clone()


def find_transformer_blocks(visual_encoder):
    """Return (names, modules) for every transformer block in the ViT."""
    names, modules = [], []
    for name, module in visual_encoder.named_modules():
        # open_clip ViT: trunk.blocks.0, trunk.blocks.1, ...
        if name.startswith('trunk.blocks.') and name.count('.') == 2:
            names.append(name)
            modules.append(module)
    if names:
        return names, modules

    # Fallback: try visual.transformer.resblocks (older open_clip)
    for name, module in visual_encoder.named_modules():
        if 'blocks' in name and name.count('.') == 1:
            names.append(name)
            modules.append(module)
    return names, modules


def cache_activations(model, dataloader, hook, device, to_numpy=True):
    """Run forward pass and return (activations_dict, labels)."""
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Caching activations'):
            images = batch['input'].to(device)
            all_labels.append(batch['target'])
            model(images)

    hook.concatenate_layer_activations()
    labels = torch.cat(all_labels, dim=0)
    if to_numpy:
        labels = labels.numpy()
        activations = {k: v.numpy() for k, v in hook.layer_activations.items()}
    else:
        activations = hook.layer_activations

    return activations, labels
