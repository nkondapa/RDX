"""
Interactive two-panel RDX cluster visualization using Plotly (RSNA wrapper).

Left panel: AlignedUMAP scatter of all images at a given layer pair, cluster members highlighted.
Right panel: On click, shows clicked image + neighbor groups with color-coded borders.

Usage:
    python -m rsna_experiments.interactive_cluster_viz --data_root /media/nkondapa/SSD2/data/RSNA
"""
import argparse
import os
import pickle as pkl

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from interactive_cluster_viz_core import (
    compute_aligned_umap,
    precompute_neighbor_data,
    precompute_matrix_data,
    precompute_ranking_data,
    precompute_classifier_labels,
    generate_html,
)
from rsna_experiments.utils import load_dataset


def load_cached_data(output_dir):
    """Load cached activations and RDX outputs."""
    cache_path = os.path.join(output_dir, 'full_cache_subset.pkl')
    print(f'Loading cached activations from {cache_path} ...')
    with open(cache_path, 'rb') as f:
        cached = pkl.load(f)

    activations = cached['activations']
    labels = cached['labels']

    # Extract CLS token (index -1, matching activation_patching.py)
    acts = {}
    layer_names = []
    for k in activations['block']:
        acts[k] = activations['block'][k][:, 0]  # (N, D)
        if isinstance(acts[k], torch.Tensor):
            acts[k] = acts[k].numpy()
        layer_names.append(k)

    opt_keys = ['post_ln', 'post_proj']
    for k in opt_keys:
        if k in activations and activations[k] is not None:
            if len(activations[k].shape) == 3:  # (N, L, D)
                acts[k] = activations[k][:, 0]
            else:
                acts[k] = activations[k]
            if isinstance(acts[k], torch.Tensor):
                acts[k] = acts[k].numpy()
            layer_names.append(k)

    # Load RDX outputs for consecutive layer pairs
    rdx_data = {}
    for i in range(len(layer_names) - 1):
        ln1, ln2 = layer_names[i], layer_names[i + 1]
        rdx_dir = os.path.join(output_dir, 'rdx_outputs', f'rdx_{ln1}_vs_{ln2}')
        outputs_path = os.path.join(rdx_dir, 'outputs.pkl')
        if os.path.exists(outputs_path):
            with open(outputs_path, 'rb') as f:
                rdx_data[(ln1, ln2)] = pkl.load(f)
        print()
    return acts, labels, layer_names, rdx_data


def save_images_as_thumbnails(data_root, output_dir, num_samples, probe_num_samples=1500, thumb_size=56):
    """Load dataset images and save as JPEG thumbnails to disk."""
    thumbs_dir = os.path.join(output_dir, 'thumbs')
    os.makedirs(thumbs_dir, exist_ok=True)

    # Use a resize-only transform (no normalization)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((thumb_size, thumb_size)),
    ])
    ds = load_dataset(data_root, transform)

    with open('inds_for_patching_and_probing.pkl', 'rb') as f:
        inds_dict = pkl.load(f)
        ptch_inds = inds_dict['ptch_inds']

    print('Saving image thumbnails...')
    for i, idx in enumerate(tqdm(ptch_inds, desc='Saving thumbnails')):
        sample = ds[idx]
        img = sample['input']  # PIL Image (after resize)
        if isinstance(img, torch.Tensor):
            img = torchvision.transforms.ToPILImage()(img)
        img.save(os.path.join(thumbs_dir, f'{i:04d}.jpg'), format='JPEG', quality=70)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', type=str,
                        default='/media/nkondapa/SSD2/data/RSNA')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/rsna_biomedclip/activation_patching')
    parser.add_argument('--num_samples', type=int, default=1500)
    parser.add_argument('--probe_num_samples', type=int, default=1500,
                        help='Must match value used in activation_patching.py')
    parser.add_argument('--K', type=int, default=12,
                        help='Number of neighbors to show')
    parser.add_argument('--K_matrix', type=int, default=12,
                        help='Number of neighbors for matrix view tab')
    parser.add_argument('--thumb_size', type=int, default=280)
    parser.add_argument('--clf_method', type=str, default='knn', choices=['knn', 'lin'],
                        help='Classifier for scatter coloring: knn or lin (linear probe)')
    parser.add_argument('--lazy_load', action='store_true',
                        help='Lazy-load large data files for faster initial page render')
    args = parser.parse_args()

    # UI sizing config — edit here or load from JSON later
    ui_config = {
        # -- Layout --
        'body_margin': 16,
        'panel_gap': 16,
        'panel_padding': 16,
        'panel_radius': 10,
        'left_flex': 3,               # *** ratio of left:right panel width
        'left_min_width': 600,        # *** scatter panel min width
        'right_flex': 2,              # *** ratio of right:left panel width
        'right_min_width': 500,       # *** neighbor panel min width
        'controls_gap': 24,
        'controls_margin_bottom': 14,
        # -- Fonts (px) --
        'font_controls': 16,
        'font_select': 15,
        'font_h2': 22,               # ** section titles
        'font_h3': 17,
        'font_legend': 14,
        'font_info': 15,
        'font_idx_label': 10,
        'font_placeholder': 16,
        'font_plot_axis_title': 16,
        'font_plot_tick': 13,
        'font_plot_general': 14,
        'font_plot_legend': 13,
        # -- Images (px) --
        'thumb_display_size': 96,     # *** neighbor thumbnail size — biggest visual impact
        'thumb_border_width': 4,
        'thumb_grid_gap': 6,
        'clicked_img_size': 180,      # *** clicked/selected image size
        'clicked_img_border': 4,
        # -- Modal (click-to-enlarge) --
        'modal_img_size': 400,        # px — enlarged image size in modal
        'modal_border': 3,
        'modal_font_size': 16,
        # -- Scatter markers --
        'marker_null_size': 6,        # ** background (null cluster) dot size
        'marker_null_opacity': 0.3,
        'marker_cluster_size': 10,    # *** colored cluster dot size — most visible scatter knob
        'marker_cluster_opacity': 0.8,
        'scatter_min_height': 600,    # *** scatter plot min height in px
        'scatter_height_offset': 160, # ** subtracted from viewport for scatter height
        # -- Legend --
        'legend_gap': 12,
        'legend_swatch': 18,
        # -- Radio buttons --
        'radio_size': 16,
        # -- Viz mode: 'single' (original) or 'transition' (side-by-side + animate) --
        'viz_mode': 'transition',
        # -- Animation (transition mode only) --
        'anim_duration_ms': 2000,        # ** playback duration in ms
        # -- Neighbor panel options --
        'show_post_spatial_nn': True,    # show post-layer spatial neighbors for convergence (10)
        # -- Comparison mode --
        'comparison_mode': 'layerwise',  # 'layerwise' = convergence/spreading, 'cross_model' = neutral R0/R1
        # -- UMAP alignment --
        'umap_mode': 'global',           # ** 'global' = align all layers with window, 'pairwise' = align each pair independently
        'K': args.K,
        'K_matrix': args.K_matrix,
    }

    # 1. Load cached data
    acts, labels, layer_names, rdx_data = load_cached_data(args.output_dir)
    N = acts[layer_names[0]].shape[0]
    print(f'Loaded {N} samples across {len(layer_names)} layers')
    print(f'Layer pairs with RDX data: {len(rdx_data)}')

    # 2. Save image thumbnails to disk
    save_images_as_thumbnails(args.data_root, args.output_dir, args.num_samples,
                              probe_num_samples=args.probe_num_samples,
                              thumb_size=args.thumb_size)

    # 3. Compute AlignedUMAP
    embeddings = compute_aligned_umap(acts, layer_names, mode=ui_config.get('umap_mode', 'pairwise'))

    # 4. Precompute neighbor data
    neighbor_data = precompute_neighbor_data(acts, layer_names, rdx_data, K=args.K)

    # 5. Precompute matrix data
    matrix_data = precompute_matrix_data(rdx_data, layer_names, K_matrix=args.K_matrix)

    # 6. Precompute ranking data
    ranking_data = precompute_ranking_data(rdx_data, layer_names, K_matrix=args.K_matrix)

    # 7. Precompute classifier labels
    clf_data = precompute_classifier_labels(acts, layer_names, labels,
                                            method=args.clf_method, K_knn=args.K)
    knn_kw = 'knn_data' if args.clf_method == 'knn' else 'lin_data'

    # 8. Generate HTML
    output_path = os.path.join(args.output_dir, 'interactive_viz.html')
    generate_html(embeddings, layer_names, rdx_data, neighbor_data,
                  'thumbs', labels, output_path, ui_config=ui_config,
                  matrix_data=matrix_data, ranking_data=ranking_data,
                  lazy_load=args.lazy_load, **{knn_kw: clf_data})


if __name__ == '__main__':
    main()
