"""
Interactive RDX cluster visualization for general cross-model comparisons.

Works with the output of generate_comparison_explanations.py, which compares
two models on arbitrary datasets.

Usage:
    python interactive_cluster_viz_general.py \
        --rdx_output_dir outputs/.../rdx_nb_lb_spectral \
        --repr_0_path outputs/.../m0_rep.pkl \
        --repr_1_path outputs/.../m1_rep.pkl \
        --image_paths outputs/.../image_paths.pkl \
        --data_root ./data/inat_subset/
"""
import argparse
import os
import pickle as pkl

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from interactive_cluster_viz_core import (
    encode_thumbnail,
    compute_aligned_umap,
    precompute_neighbor_data,
    precompute_matrix_data,
    precompute_ranking_data,
    precompute_classifier_labels,
    generate_html,
)


def load_comparison_data(rdx_output_dir, repr_0_path, repr_1_path,
                         repr_0_name='repr_0', repr_1_name='repr_1'):
    """Load representations and RDX outputs from a general cross-model comparison.

    Returns:
        acts: {repr_0_name: (N, D) ndarray, repr_1_name: (N, D) ndarray}
        labels: None (no labels available by default)
        layer_names: [repr_0_name, repr_1_name]
        rdx_data: {(repr_0_name, repr_1_name): normalized_output}
    """
    # Load representations
    print(f'Loading representations from {repr_0_path} and {repr_1_path} ...')
    with open(repr_0_path, 'rb') as f:
        r0 = pkl.load(f)
    with open(repr_1_path, 'rb') as f:
        r1 = pkl.load(f)

    # Convert to numpy if torch tensors
    if isinstance(r0, torch.Tensor):
        r0 = r0.numpy()
    if isinstance(r1, torch.Tensor):
        r1 = r1.numpy()
    r0 = np.array(r0, dtype=np.float32)
    r1 = np.array(r1, dtype=np.float32)

    # Load RDX outputs
    outputs_path = os.path.join(rdx_output_dir, 'outputs.pkl')
    print(f'Loading RDX outputs from {outputs_path} ...')
    with open(outputs_path, 'rb') as f:
        rdx_output = pkl.load(f)

    # Normalize cluster_dict keys: 'am_10' -> '10', 'am_01' -> '01'
    cluster_dict = rdx_output.get('cluster_dict', {})
    normalized_cd = {}
    for k, v in cluster_dict.items():
        norm_key = k.replace('am_', '') if k.startswith('am_') else k
        normalized_cd[norm_key] = v
    rdx_output['cluster_dict'] = normalized_cd

    acts = {repr_0_name: r0, repr_1_name: r1}
    layer_names = [repr_0_name, repr_1_name]
    rdx_data = {(repr_0_name, repr_1_name): rdx_output}

    return acts, None, layer_names, rdx_data


def load_thumbnails_from_paths(image_paths_file, data_root=None, thumb_size=56):
    """Load image thumbnails from a pkl file containing image paths.

    The pkl file can contain either:
        - A dict with 'paths' (list of paths) and optionally 'data_root' (str)
        - A plain list of image paths

    Args:
        image_paths_file: Path to pkl file with image paths.
        data_root: Optional override for the data root prefix.
        thumb_size: Thumbnail size in pixels.

    Returns:
        List of base64-encoded JPEG thumbnail strings.
    """
    print(f'Loading image paths from {image_paths_file} ...')
    with open(image_paths_file, 'rb') as f:
        data = pkl.load(f)

    if isinstance(data, dict):
        paths = data['paths']
        if data_root is None:
            data_root = data.get('data_root', '')
        labels = data.get('labels', None)
    else:
        paths = data
        if data_root is None:
            data_root = ''
        labels = None

    print(f'Encoding {len(paths)} thumbnails...')
    thumbnails = []
    for p in tqdm(paths, desc='Encoding thumbnails'):
        full_path = os.path.join(data_root, p) if data_root else p
        img = Image.open(full_path).convert('RGB')
        img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
        thumbnails.append(encode_thumbnail(img))

    return thumbnails, labels


def main():
    parser = argparse.ArgumentParser(
        description='Interactive RDX cluster visualization for general cross-model comparisons.')
    parser.add_argument('--rdx_output_dir', type=str, required=True,
                        help='Path to RDX output directory (e.g. .../rdx_nb_lb_spectral/)')
    parser.add_argument('--repr_0_path', type=str, required=True,
                        help='Path to model 0 representation pkl file')
    parser.add_argument('--repr_1_path', type=str, required=True,
                        help='Path to model 1 representation pkl file')
    parser.add_argument('--repr_0_name', type=str, default='Model 0',
                        help='Display name for model 0')
    parser.add_argument('--repr_1_name', type=str, default='Model 1',
                        help='Display name for model 1')
    parser.add_argument('--image_paths', type=str, default=None,
                        help='Path to pkl file with image paths for thumbnails')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Optional root prefix for relative image paths')
    parser.add_argument('--labels_path', type=str, default=None,
                        help='Optional path to labels pkl/npy file')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output HTML file path (default: <rdx_output_dir>/interactive_viz.html)')
    parser.add_argument('--thumb_size', type=int, default=96)
    parser.add_argument('--K', type=int, default=12,
                        help='Number of neighbors to show')
    parser.add_argument('--K_matrix', type=int, default=12,
                        help='Number of neighbors for matrix view tab')
    parser.add_argument('--clf_method', type=str, default='knn', choices=['knn', 'lin'],
                        help='Classifier for scatter coloring: knn or lin (linear probe)')
    args = parser.parse_args()

    # UI config
    ui_config = {
        'body_margin': 16,
        'panel_gap': 16,
        'panel_padding': 16,
        'panel_radius': 10,
        'left_flex': 3,
        'left_min_width': 600,
        'right_flex': 2,
        'right_min_width': 500,
        'controls_gap': 24,
        'controls_margin_bottom': 14,
        'font_controls': 16,
        'font_select': 15,
        'font_h2': 22,
        'font_h3': 17,
        'font_legend': 14,
        'font_info': 15,
        'font_idx_label': 10,
        'font_placeholder': 16,
        'font_plot_axis_title': 16,
        'font_plot_tick': 13,
        'font_plot_general': 14,
        'font_plot_legend': 13,
        'thumb_display_size': 96,
        'thumb_border_width': 4,
        'thumb_grid_gap': 6,
        'clicked_img_size': 180,
        'clicked_img_border': 4,
        # -- Modal (click-to-enlarge) --
        'modal_img_size': 400,        # px — enlarged image size in modal
        'modal_border': 3,
        'modal_font_size': 16,
        # -- Scatter markers --
        'marker_null_size': 6,
        'marker_null_opacity': 0.3,
        'marker_cluster_size': 10,
        'marker_cluster_opacity': 0.8,
        'scatter_min_height': 600,
        'scatter_height_offset': 160,
        'legend_gap': 12,
        'legend_swatch': 18,
        'radio_size': 16,
        'viz_mode': 'transition',
        'anim_duration_ms': 2000,
        'show_post_spatial_nn': True,
        'umap_mode': 'pairwise',
        'K': args.K,
        'K_matrix': args.K_matrix,
    }

    # 1. Load data
    acts, labels, layer_names, rdx_data = load_comparison_data(
        args.rdx_output_dir, args.repr_0_path, args.repr_1_path,
        repr_0_name=args.repr_0_name, repr_1_name=args.repr_1_name)
    N = acts[layer_names[0]].shape[0]
    print(f'Loaded {N} samples, comparing {layer_names[0]} vs {layer_names[1]}')

    # # 2. Load labels if provided
    # if args.labels_path is not None:
    #     print(f'Loading labels from {args.labels_path} ...')
    #     if args.labels_path.endswith('.npy'):
    #         labels = np.load(args.labels_path)
    #     else:
    #         with open(args.labels_path, 'rb') as f:
    #             labels = pkl.load(f)
    #     if isinstance(labels, torch.Tensor):
    #         labels = labels.numpy()
    #     labels = np.array(labels)
    #
    # # Default labels to zeros if not available


    # 3. Load thumbnails
    if args.image_paths is not None:
        thumbnails, labels = load_thumbnails_from_paths(args.image_paths, data_root=args.data_root,
                                                thumb_size=args.thumb_size)
    else:
        # Generate placeholder thumbnails (gray squares)
        print('No image paths provided, using placeholder thumbnails...')
        placeholder = Image.new('RGB', (args.thumb_size, args.thumb_size), (128, 128, 128))
        placeholder_b64 = encode_thumbnail(placeholder)
        thumbnails = [placeholder_b64] * N

    if labels is None:
        labels = np.zeros(N, dtype=int)

    # 4. Compute AlignedUMAP
    embeddings = compute_aligned_umap(acts, layer_names, mode=ui_config.get('umap_mode', 'pairwise'))

    # 5. Precompute neighbor data
    neighbor_data = precompute_neighbor_data(acts, layer_names, rdx_data, K=args.K)

    # 6. Precompute matrix data
    matrix_data = precompute_matrix_data(rdx_data, layer_names, K_matrix=args.K_matrix)

    # 7. Precompute ranking data
    ranking_data = precompute_ranking_data(rdx_data, layer_names, K_matrix=args.K_matrix)

    # 8. Precompute classifier labels
    clf_data = precompute_classifier_labels(acts, layer_names, labels,
                                            method=args.clf_method, K_knn=args.K)
    knn_kw = 'knn_data' if args.clf_method == 'knn' else 'lin_data'

    # 9. Generate HTML
    output_path = args.output_path or os.path.join(args.rdx_output_dir, 'interactive_viz.html')
    generate_html(embeddings, layer_names, rdx_data, neighbor_data,
                  thumbnails, labels, output_path, ui_config=ui_config,
                  matrix_data=matrix_data, ranking_data=ranking_data,
                  **{knn_kw: clf_data})


if __name__ == '__main__':
    main()
