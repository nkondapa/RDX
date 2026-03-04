"""
Activation patching + RDX experiments on BiomedCLIP (RSNA Pneumonia).

Usage:
    python -m rsna_experiments.analyze_biomedclip [--data_root PATH] [--output_dir PATH]
                                                   [--batch_size N] [--device DEV]
"""
import copy
import os
import pickle as pkl
from functools import partial

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import colors
import open_clip.timm_model
from src.utils.hooks import ActivationHookV2
from src.rdx import RDX
from rsna_experiments.utils import (
    parse_args, load_model, load_dataset,
    find_transformer_blocks,
)
from src.cka import CKA
from src.utils.plotting_helper import finish_plot
from linear_probes import run_probes_full, run_knn_probes_full, evaluate_probes


# ── Patching helpers ─────────────────────────────────────────────────────────
def patch_hook(x, head_index, patch_val, num_heads, head_dim):
    B, S, C = x.shape
    x = x.reshape(B, S, num_heads, head_dim)
    x[:, :, head_index, :] = patch_val
    x = x.reshape(B, S, C)
    return x


def topk_neighbor_similarity(m0_embs, m1_embs, k=16):
    m0_neighbor_mat = torch.cdist(m0_embs, m0_embs)
    m1_neighbor_mat = torch.cdist(m1_embs, m1_embs)
    m0_nn = m0_neighbor_mat.argsort(dim=1)
    m1_nn = m1_neighbor_mat.argsort(dim=1)

    isti = []
    for i in range(m0_nn.shape[0]):
        topk_m0 = set(m0_nn[i, 1:k + 1].flatten().tolist())
        topk_m1 = set(m1_nn[i, 1:k + 1].flatten().tolist())
        intersection = len(topk_m0.intersection(topk_m1))
        isti.append(intersection / k)
    return torch.mean(torch.tensor(isti)).item()


def cka_similarity(m0_embs, m1_embs):
    cka = CKA(debiased=True)
    similarity = cka.linear_CKA(m0_embs.cpu().numpy(), m1_embs.cpu().numpy())
    return similarity.item()


def relative_cluster_neighborhood_distance_delta(patched_embs, rdx_data, k=16, n_clusters=None):
    patched_dist = torch.cdist(patched_embs, patched_embs).argsort(-1).argsort(-1).type(torch.float32)
    target_dist = rdx_data['graph_dict']['r1_dm']

    scores = []
    selectivity_scores = []
    for direct in ['01', '10']:
        array = rdx_data['cluster_dict'][f'{direct}']['cluster_labels']
        existing_labels = set(np.unique(array)) - {0}  # non-null labels that exist

        # Iterate over fixed range 1..n_clusters so output length is always consistent
        cluster_range = range(1, n_clusters + 2) if n_clusters is not None else sorted(existing_labels)
        for ci in cluster_range:
            if ci not in existing_labels:
                # Cluster was merged into null — output 0
                scores.append(0.0)
                selectivity_scores.append(0.0)
                continue
            sel_inds = (array == ci).nonzero()[0]
            not_sel_inds = (array != ci).nonzero()[0]
            patched_sel_dist = patched_dist[sel_inds][:, sel_inds].mean()
            target_sel_dist = target_dist[sel_inds][:, sel_inds].mean()
            patched_nsel_dist = patched_dist[not_sel_inds][:, not_sel_inds]
            target_nsel_dist = target_dist[not_sel_inds][:, not_sel_inds]
            non_cluster_change = torch.abs(target_nsel_dist - patched_nsel_dist).mean() + 1e-8
            if direct == '01':
                rdx_nb_dd = -1 * (patched_sel_dist - target_sel_dist)
            else:
                rdx_nb_dd = (patched_sel_dist - target_sel_dist)
            selectivity_score = rdx_nb_dd / non_cluster_change
            selectivity_scores.append(selectivity_score.item())
            scores.append(rdx_nb_dd.item())

    return scores, selectivity_scores


def activation_patching(replaced_attn, block_modules, all_layer_names, activations, labels,
                        data, probe_results, rdx_params, output_path, topk=16):
    ra_dict = dict(replaced_attn)
    batch_size = 64
    metrics = {'cka': [], 'topkns': [],
               'rdx_clnb_dist_delta': [], 'rdx_clnb_dist_delta_selectivity': [],
               'rdx_topk_nb_dist_delta': [], 'rdx_topk_nb_dist_delta_selectivity': [],
               'knn_acc': [], 'linear_acc': []}
    probe_predictions = {'knn': [], 'linear': []}
    rdx_sampling = {'within_cluster_sampling': False, 'method': 'affinity'}
    for bi in range(len(all_layer_names) - 1):
        input_layer = all_layer_names[bi]
        target_layer = all_layer_names[bi + 1]
        name = f'{target_layer}.attn'
        m = ra_dict[name]
        for metric in metrics.keys():
            metrics[metric].append([])
        for metric in probe_predictions.keys():
            probe_predictions[metric].append([])

        for head_index in range(m.num_heads):
            patch_val = activations['attn_head'][name][:, head_index, :]
            m.register_hook('attn_output', partial(patch_hook, head_index=head_index, patch_val=patch_val,
                                                   num_heads=m.num_heads, head_dim=m.head_dim))
            next_block_acts = []
            with torch.no_grad():
                for bsi in range(0, len(activations['block'][input_layer]), batch_size):
                    batch_block_act = activations['block'][input_layer][bsi:bsi + batch_size].cuda()
                    next_block_act = block_modules[bi](batch_block_act).cpu()
                    next_block_acts.append(next_block_act)
                    torch.cuda.empty_cache()
            patched_acts = torch.cat(next_block_acts)
            score = 1 - topk_neighbor_similarity(patched_acts[:, 0], activations['block'][target_layer][:, 0],
                                                 k=topk)
            cka_score = 1 - cka_similarity(patched_acts[:, 0], activations['block'][target_layer][:, 0])
            rdx_clnb_dd, rclndd_ss = relative_cluster_neighborhood_distance_delta(
                patched_acts[:, 0],
                data['output_dict'][(input_layer, target_layer)],
                k=topk, n_clusters=rdx_params['n_clusters'])
            rdx_topk_dd, rtopk_ss = topk_cluster_neighborhood_distance_delta(
                patched_acts[:, 0],
                data['output_dict'][(input_layer, target_layer)],
                k=topk, n_clusters=rdx_params['n_clusters'],
                rdx_sampling=rdx_sampling)
            act_dict = {target_layer: patched_acts[:, 0]}
            knn_result = evaluate_probes(probe_results['knn_train'], act_dict, np.array(labels), verbose=False)
            linear_result = evaluate_probes(probe_results['linear_train'], act_dict, np.array(labels), verbose=False)

            metrics['cka'][-1].append(cka_score)
            metrics['topkns'][-1].append(score)
            metrics['rdx_clnb_dist_delta'][-1].append(rdx_clnb_dd)
            metrics['rdx_clnb_dist_delta_selectivity'][-1].append(rclndd_ss)
            metrics['rdx_topk_nb_dist_delta'][-1].append(rdx_topk_dd)
            metrics['rdx_topk_nb_dist_delta_selectivity'][-1].append(rtopk_ss)
            metrics['knn_acc'][-1].append(knn_result[target_layer]['accuracy'])
            metrics['linear_acc'][-1].append(linear_result[target_layer]['accuracy'])
            probe_predictions['knn'][-1].append(knn_result[target_layer]['predictions'])
            probe_predictions['linear'][-1].append(linear_result[target_layer]['predictions'])

            print(
                f'Block {target_layer} head {head_index} | topkns score: {score:.4f} | cka score: {cka_score:.4f} '
                f'| rdxnbdd score {np.abs(rdx_clnb_dd).sum():.4f} | rdxnbddss score {np.abs(rclndd_ss).sum():.4f}')
            m.clear_hooks()
        with open(output_path, 'wb') as f:
            pkl.dump(dict(metrics=metrics, probe_predictions=probe_predictions), f)

    return metrics, probe_predictions


def topk_cluster_neighborhood_distance_delta(patched_embs, rdx_data, k=16, n_clusters=None,
                                             rdx_sampling=None):
    rdx_sampling = rdx_sampling or {}
    wc_sampling = rdx_sampling.get('within_cluster_sampling', True)
    samp_method = rdx_sampling.get('method', 'spatial')

    patched_dist = torch.cdist(patched_embs, patched_embs)
    patched_dist = patched_dist.argsort(-1).argsort(-1).type(torch.float32)
    patched_dist = (patched_dist + patched_dist.T) / 2

    scores = []
    selectivity_scores = []
    n_samples = patched_embs.shape[0]

    for direct in ['01', '10']:
        # Clustered representation determines the neighborhood structure
        if direct == '10':
            clustered_dm = rdx_data['graph_dict']['r1_dm']
        else:
            clustered_dm = rdx_data['graph_dict']['r0_dm']
        target_dm = clustered_dm

        array = rdx_data['cluster_dict'][f'{direct}']['cluster_labels']
        unique_labels = np.unique(array)

        am_mat = None
        if samp_method == 'affinity':
            am_mat = rdx_data['graph_dict'].get(f'am_{direct}')
            if am_mat is not None and not isinstance(am_mat, torch.Tensor):
                am_mat = torch.tensor(am_mat, dtype=torch.float32)

        if not wc_sampling:
            # Per-sample mode: compute delta for every sample
            for idx in range(n_samples):
                if samp_method == 'spatial':
                    d = clustered_dm[idx].clone()
                    d[idx] = float('inf')
                    topk_nn = d.argsort()[:k]
                    topk_nn = topk_nn[d[topk_nn] < float('inf')]
                elif samp_method == 'affinity':
                    a = am_mat[idx].clone()
                    a[idx] = float('-inf')
                    topk_nn = a.argsort(descending=True)[:k]
                    topk_nn = topk_nn[a[topk_nn] > float('-inf')]

                if len(topk_nn) == 0:
                    scores.append(0.0)
                    selectivity_scores.append(0.0)
                    continue

                delta = (patched_dist[topk_nn][:, topk_nn] - target_dm[topk_nn][:, topk_nn]).mean().item()

                # Selectivity: delta to top-K vs. mean change to non-top-K points
                topk_set = set(topk_nn.tolist()) | {idx}
                non_topk_mask = torch.ones(n_samples, dtype=torch.bool)
                for j in topk_set:
                    non_topk_mask[j] = False
                non_topk_change = torch.abs(
                    patched_dist[idx, non_topk_mask] - target_dm[idx, non_topk_mask]).mean().item() + 1e-8

                if direct == '01':
                    delta = -1 * delta

                scores.append(delta)
                selectivity_score = delta / non_topk_change
                selectivity_scores.append(selectivity_score)
        else:
            # Within-cluster mode: per-cluster averaged scores
            existing_labels = set(unique_labels) - {0}
            cluster_mask_all = {ci: (array == ci) for ci in unique_labels}
            cluster_range = range(1, n_clusters + 2) if n_clusters is not None else sorted(existing_labels)

            for ci in cluster_range:
                if ci not in existing_labels:
                    scores.append(0.0)
                    selectivity_scores.append(0.0)
                    continue
                sel_inds = (array == ci).nonzero()[0]
                not_sel_inds = (array != ci).nonzero()[0]
                c_mask = cluster_mask_all[ci]

                cluster_deltas = []
                for idx in sel_inds:
                    if samp_method == 'spatial':
                        cl_d = clustered_dm[idx].clone()
                        cl_d[idx] = float('inf')
                        cl_d[~torch.tensor(c_mask)] = float('inf')
                        topk_nn = cl_d.argsort()[:k]
                        topk_nn = topk_nn[cl_d[topk_nn] < float('inf')]
                    elif samp_method == 'affinity':
                        cl_a = am_mat[idx].clone()
                        cl_a[idx] = float('-inf')
                        cl_a[~torch.tensor(c_mask)] = float('-inf')
                        topk_nn = cl_a.argsort(descending=True)[:k]
                        topk_nn = topk_nn[cl_a[topk_nn] > float('-inf')]

                    if len(topk_nn) == 0:
                        continue

                    patched_dist_i = patched_dist[idx, topk_nn].mean()
                    target_dist_i = target_dm[idx, topk_nn].mean()
                    cluster_deltas.append((patched_dist_i - target_dist_i).item())

                if len(cluster_deltas) == 0:
                    scores.append(0.0)
                    selectivity_scores.append(0.0)
                    continue

                mean_delta = np.mean(cluster_deltas)

                # Non-cluster change for selectivity
                patched_nsel_dist = patched_dist[not_sel_inds][:, not_sel_inds]
                target_nsel_dist = target_dm[not_sel_inds][:, not_sel_inds]
                non_cluster_change = torch.abs(target_nsel_dist - patched_nsel_dist).mean().item() + 1e-8

                if direct == '01':
                    rdx_nb_dd = -1 * mean_delta
                else:
                    rdx_nb_dd = mean_delta

                selectivity_score = rdx_nb_dd / non_cluster_change
                selectivity_scores.append(selectivity_score)
                scores.append(rdx_nb_dd)

    return scores, selectivity_scores


def plot_probe_accuracies(probe_results, args, fig_output_dir=None, show=True, save=False):
    """Plot KNN and linear probe test accuracies across layers (blog-post style)."""
    import matplotlib.ticker as mticker

    # Extract layer names and accuracies
    layer_keys = list(probe_results['knn_test'].keys())
    knn_acc = [probe_results['knn_test'][k]['accuracy'] for k in layer_keys]
    lin_acc = [probe_results['linear_test'][k]['accuracy'] for k in layer_keys]

    # Clean layer names for x-axis labels
    def short_name(k):
        if k.startswith('pre_'):
            return 'pre'
        elif 'blocks.' in k:
            return k.split('.')[-1]
        elif k == 'post_ln':
            return 'LN'
        elif k == 'post_proj':
            return 'proj'
        return k

    x_labels = [short_name(k) for k in layer_keys]
    x = np.arange(len(layer_keys))

    # Blog-style colors
    blue = '#2d5a8e'
    coral = '#c0584f'
    bg = '#fafaf8'
    text = '#1a1a1a'
    muted = '#6b6b6b'
    border = '#e2e2de'

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.plot(x, knn_acc, color=blue, marker='o', markersize=5, linewidth=2, label=f'KNN (k={args.topk})')

    ax.set_xlabel('Layer', fontfamily='Source Sans 3', fontsize=12, color=text)
    ax.set_ylabel('Test Accuracy', fontfamily='Source Sans 3', fontsize=12, color=text)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontfamily='Source Sans 3', fontsize=10, color=muted)
    ax.tick_params(axis='y', labelsize=10, labelcolor=muted)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    # Minimal spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(border)
    ax.spines['bottom'].set_color(border)
    ax.tick_params(colors=muted)
    ax.grid(axis='y', color=border, linewidth=0.5, alpha=0.7)

    ax.legend(frameon=False, fontsize=10, loc='lower right',
              prop={'family': 'Source Sans 3'})

    plt.tight_layout()
    if save and fig_output_dir:
        fig.savefig(os.path.join(fig_output_dir, 'probe_accuracies.png'), dpi=200,
                    facecolor=fig.get_facecolor(), bbox_inches='tight')
        fig.savefig(os.path.join(fig_output_dir, 'probe_accuracies.svg'),
                    facecolor=fig.get_facecolor(), bbox_inches='tight')
    if show:
        plt.show()


def measure_cluster_properties(rdx_params, data, probe_results, labels, probe_method='knn',
                               fig_output_dir=None, show=True, save=False):
    num_perms = 1000
    n_clusters = rdx_params['n_clusters'] + 1
    rng = np.random.default_rng(seed=42)
    nc = np.zeros(shape=(2, len(data['output_dict']), n_clusters + 1, 2))
    # Per-cluster enrichment: (rdx_rate, random_mean, random_std, p_value) for each cluster
    cluster_enrichment = np.zeros(shape=(2, len(data['output_dict']), n_clusters + 1, 4))
    pred_change_freq = []
    pred_change_count = []
    for ki, key in enumerate(data['output_dict']):
        cl_labels0 = data['output_dict'][key]['cluster_dict']['01']['cluster_labels']
        cl_labels1 = data['output_dict'][key]['cluster_dict']['10']['cluster_labels']
        prev_preds = probe_results[f'{probe_method}_test'][key[0]]['predictions']
        preds = probe_results[f'{probe_method}_test'][key[1]]['predictions']
        pred_change = prev_preds != preds
        pred_change_freq.append(pred_change.sum().item() / len(pred_change))
        pred_change_count.append(pred_change.sum().item())

        for i, cl_labels in enumerate([cl_labels0, cl_labels1]):
            unique_labels = np.unique(cl_labels)
            cluster_sizes = {int(labi): int((cl_labels == labi).sum()) for labi in unique_labels}
            n = len(cl_labels)

            for labi in unique_labels:
                mask = cl_labels == labi
                nc[i, ki, labi] = np.array([pred_change[mask].sum().item(), mask.sum()])

            # Permutation test: draw random clusters of the same sizes, measure per-cluster rates
            random_rates = np.zeros(shape=(num_perms, n_clusters + 1))
            for pi in range(num_perms):
                perm = rng.permutation(n)
                offset = 0
                for labi in unique_labels:
                    size = cluster_sizes[labi]
                    perm_mask = perm[offset:offset + size]
                    random_rates[pi, labi] = pred_change[perm_mask].sum() / size
                    offset += size

            direction_str = '01' if i == 0 else '10'
            for labi in unique_labels:
                size = cluster_sizes[labi]
                rdx_rate = nc[i, ki, labi, 0] / size
                rand_mean = random_rates[:, labi].mean()
                rand_std = random_rates[:, labi].std()
                p_value = (random_rates[:, labi] >= rdx_rate).sum() / num_perms
                cluster_enrichment[i, ki, labi] = [rdx_rate, rand_mean, rand_std, p_value]
                print(f'  {key} dir={direction_str} cluster={labi} | rdx_rate={rdx_rate:.4f} '
                      f'random={rand_mean:.4f}+/-{rand_std:.4f} '
                      f'p={p_value:.4f} (n={size})')

    pred_change_freq = np.array(pred_change_freq)
    layer_pair_keys = list(data['output_dict'].keys())
    layer_pair_labels = [f'{k[0].split(".")[-1]}-{k[1].split(".")[-1]}' for k in layer_pair_keys]
    n_pairs = len(layer_pair_keys)

    # Build a mask of which clusters actually exist per (direction, layer_pair)
    # cluster_exists[dir_idx, pair_idx, cluster_idx_0based] = True if non-null cluster exists
    cluster_exists = np.zeros((2, n_pairs, n_clusters), dtype=bool)
    for ki, key in enumerate(data['output_dict']):
        for i, direction_str in enumerate(['01', '10']):
            cl_labels = data['output_dict'][key]['cluster_dict'][direction_str]['cluster_labels']
            for labi in np.unique(cl_labels):
                if labi > 0:
                    cluster_exists[i, ki, labi - 1] = True

    # Count actual non-null clusters per (direction, layer_pair)
    n_actual_clusters = cluster_exists.sum(axis=2)  # (2, n_pairs)

    def _mask_heatmap(arr_2d, exists_2d):
        """Set cells to NaN where cluster doesn't exist, for clean heatmap display."""
        masked = arr_2d.astype(float).copy()
        masked[~exists_2d] = np.nan  # arr is (n_pairs, n_clusters), exists is (n_pairs, n_clusters)
        return masked

    # --- Plot 1: p-value heatmap (directions x clusters) across layer pairs ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, direction_str in enumerate(['01', '10']):
        pvals = cluster_enrichment[i, :, 1:, 3]  # (n_pairs, n_clusters)
        pvals_masked = _mask_heatmap(pvals, cluster_exists[i])
        im = axes[i].imshow(pvals_masked.T, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[i].set_title(f'Enrichment p-value (dir={direction_str})')
        axes[i].set_xlabel('Layer Pair')
        axes[i].set_ylabel('Cluster')
        axes[i].set_xticks(range(n_pairs))
        axes[i].set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
        axes[i].set_yticks(range(n_clusters))
        axes[i].set_yticklabels([f'C{c + 1}' for c in range(n_clusters)])
        fig.colorbar(im, ax=axes[i], label='p-value')
        for ki_idx in range(n_pairs):
            for ci in range(n_clusters):
                if cluster_exists[i, ki_idx, ci] and pvals[ki_idx, ci] < 0.05:
                    axes[i].text(ki_idx, ci, '*', ha='center', va='center', color='black', fontsize=12,
                                 fontweight='bold')
    plt.tight_layout()
    finish_plot(show, save, os.path.join(fig_output_dir, 'cluster_enrichment_pvalues.png') if fig_output_dir else None,
                fig=fig)

    # --- Plot 2: RDX rate vs random baseline per layer pair (non-null clusters aggregated) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, direction_str in enumerate(['01', '10']):
        rdx_rates = cluster_enrichment[i, :, 1:, 0]  # (n_pairs, n_clusters)
        rand_means = cluster_enrichment[i, :, 1:, 1]
        rand_stds = cluster_enrichment[i, :, 1:, 2]
        cluster_sizes = nc[i, :, 1:, 1]  # (n_pairs, n_clusters)

        # Size-weighted average across clusters per layer pair
        total_sizes = cluster_sizes.sum(axis=1, keepdims=True)
        weights = np.divide(cluster_sizes, total_sizes, where=total_sizes > 0,
                            out=np.zeros_like(cluster_sizes))
        avg_rdx_rate = (rdx_rates * weights).sum(axis=1)
        avg_rand_mean = (rand_means * weights).sum(axis=1)
        avg_rand_std = (rand_stds * weights).sum(axis=1)

        x = np.arange(n_pairs)
        axes[i].plot(x, avg_rdx_rate, 'o-', label='RDX cluster rate', color='tab:red')
        axes[i].plot(x, avg_rand_mean, 's--', label='Random baseline', color='tab:blue')
        axes[i].fill_between(x, avg_rand_mean - avg_rand_std, avg_rand_mean + avg_rand_std,
                             alpha=0.2, color='tab:blue')
        axes[i].plot(x, pred_change_freq, '^:', label='Overall rate', color='tab:gray')
        axes[i].set_title(f'Pred-change rate in clusters (dir={direction_str})')
        axes[i].set_xlabel('Layer Pair')
        axes[i].set_ylabel('Pred-change rate')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
        axes[i].legend(fontsize=8)
    plt.tight_layout()
    finish_plot(show, save, os.path.join(fig_output_dir, 'cluster_pred_change_rate.png') if fig_output_dir else None,
                fig=fig)

    # --- Plot 3: Per-cluster enrichment (rdx_rate - random_mean) heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, direction_str in enumerate(['01', '10']):
        enrichment = cluster_enrichment[i, :, 1:, 0] - cluster_enrichment[i, :, 1:, 1]  # (n_pairs, n_clusters)
        enrichment_masked = _mask_heatmap(enrichment, cluster_exists[i])
        vmax = np.nanmax(np.abs(enrichment_masked))
        im = axes[i].imshow(enrichment_masked.T, cmap='bwr', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[i].set_title(f'Enrichment (rdx - random) (dir={direction_str})')
        axes[i].set_xlabel('Layer Pair')
        axes[i].set_ylabel('Cluster')
        axes[i].set_xticks(range(n_pairs))
        axes[i].set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
        axes[i].set_yticks(range(n_clusters))
        axes[i].set_yticklabels([f'C{c + 1}' for c in range(n_clusters)])
        fig.colorbar(im, ax=axes[i], label='Rate difference')
        for ki_idx in range(n_pairs):
            for ci in range(n_clusters):
                if cluster_exists[i, ki_idx, ci] and cluster_enrichment[i, ki_idx, ci + 1, 3] < 0.05:
                    axes[i].text(ki_idx, ci, '*', ha='center', va='center', color='black', fontsize=12,
                                 fontweight='bold')
    plt.tight_layout()
    finish_plot(show, save, os.path.join(fig_output_dir, 'cluster_enrichment_diff.png') if fig_output_dir else None,
                fig=fig)

    # --- Plot 4: Fraction of significant clusters per layer pair ---
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(n_pairs)
    width = 0.35
    for i, direction_str in enumerate(['01', '10']):
        pvals = cluster_enrichment[i, :, 1:, 3]  # (n_pairs, n_clusters)
        # Only count significant among clusters that actually exist
        sig_count = np.array([(pvals[ki, cluster_exists[i, ki]] < 0.05).sum() for ki in range(n_pairs)])
        denom = np.maximum(n_actual_clusters[i], 1)
        frac_sig = sig_count / denom
        ax.bar(x + i * width - width / 2, frac_sig, width, label=f'dir={direction_str}')
    ax.set_xlabel('Layer Pair')
    ax.set_ylabel('Fraction of clusters with p < 0.05')
    ax.set_title('Fraction of significantly enriched clusters per layer pair')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.set_ylim([0, 1])
    plt.tight_layout()
    finish_plot(show, save,
                os.path.join(fig_output_dir, 'fraction_significant_clusters.png') if fig_output_dir else None, fig=fig)

    # --- Class homogeneity analysis: are RDX clusters more class-selective than random? ---
    labels_np = np.array(labels)
    unique_classes = np.unique(labels_np)
    n_classes = len(unique_classes)
    # class_homogeneity[dir, layer_pair, cluster] = (rdx_majority_frac, random_mean, random_std, p_value)
    class_homogeneity = np.zeros(shape=(2, len(data['output_dict']), n_clusters + 1, 4))
    overall_majority_frac = max([(labels_np == c).sum() for c in unique_classes]) / len(labels_np)

    for ki, key in enumerate(data['output_dict']):
        cl_labels0 = data['output_dict'][key]['cluster_dict']['01']['cluster_labels']
        cl_labels1 = data['output_dict'][key]['cluster_dict']['10']['cluster_labels']

        for i, cl_labels in enumerate([cl_labels0, cl_labels1]):
            unique_cl = np.unique(cl_labels)
            cluster_sizes = {int(labi): int((cl_labels == labi).sum()) for labi in unique_cl}
            n = len(cl_labels)

            # Permutation test for majority-class fraction
            random_maj_frac = np.zeros(shape=(num_perms, n_clusters + 1))
            for pi in range(num_perms):
                perm = rng.permutation(n)
                offset = 0
                for labi in unique_cl:
                    size = cluster_sizes[labi]
                    perm_inds = perm[offset:offset + size]
                    perm_labels = labels_np[perm_inds]
                    class_counts = np.array([(perm_labels == c).sum() for c in unique_classes])
                    random_maj_frac[pi, labi] = class_counts.max() / size
                    offset += size

            direction_str = '01' if i == 0 else '10'
            for labi in unique_cl:
                mask = cl_labels == labi
                size = cluster_sizes[labi]
                cluster_labels_subset = labels_np[mask]
                class_counts = np.array([(cluster_labels_subset == c).sum() for c in unique_classes])
                rdx_maj_frac = class_counts.max() / size
                rand_mean = random_maj_frac[:, labi].mean()
                rand_std = random_maj_frac[:, labi].std()
                p_value = (random_maj_frac[:, labi] >= rdx_maj_frac).sum() / num_perms
                class_homogeneity[i, ki, labi] = [rdx_maj_frac, rand_mean, rand_std, p_value]
                print(f'  {key} dir={direction_str} cluster={labi} | maj_frac={rdx_maj_frac:.4f} '
                      f'random={rand_mean:.4f}+/-{rand_std:.4f} '
                      f'p={p_value:.4f} (n={size})')

    # --- Plot 5: Class homogeneity p-value heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, direction_str in enumerate(['01', '10']):
        pvals = class_homogeneity[i, :, 1:, 3]  # (n_pairs, n_clusters)
        pvals_masked = _mask_heatmap(pvals, cluster_exists[i])
        im = axes[i].imshow(pvals_masked.T, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[i].set_title(f'Class homogeneity p-value (dir={direction_str})')
        axes[i].set_xlabel('Layer Pair')
        axes[i].set_ylabel('Cluster')
        axes[i].set_xticks(range(n_pairs))
        axes[i].set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
        axes[i].set_yticks(range(n_clusters))
        axes[i].set_yticklabels([f'C{c + 1}' for c in range(n_clusters)])
        fig.colorbar(im, ax=axes[i], label='p-value')
        for ki_idx in range(n_pairs):
            for ci in range(n_clusters):
                if cluster_exists[i, ki_idx, ci] and pvals[ki_idx, ci] < 0.05:
                    axes[i].text(ki_idx, ci, '*', ha='center', va='center', color='black', fontsize=12,
                                 fontweight='bold')
    plt.tight_layout()
    finish_plot(show, save, os.path.join(fig_output_dir, 'class_homogeneity_pvalues.png') if fig_output_dir else None,
                fig=fig)

    # --- Plot 6: Majority-class fraction: RDX vs random baseline ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, direction_str in enumerate(['01', '10']):
        rdx_maj = class_homogeneity[i, :, 1:, 0]  # (n_pairs, n_clusters)
        rand_maj_mean = class_homogeneity[i, :, 1:, 1]
        rand_maj_std = class_homogeneity[i, :, 1:, 2]
        cluster_sizes_arr = nc[i, :, 1:, 1]

        total_sizes = cluster_sizes_arr.sum(axis=1, keepdims=True)
        weights = np.divide(cluster_sizes_arr, total_sizes, where=total_sizes > 0,
                            out=np.zeros_like(cluster_sizes_arr))
        avg_rdx_maj = (rdx_maj * weights).sum(axis=1)
        avg_rand_mean = (rand_maj_mean * weights).sum(axis=1)
        avg_rand_std = (rand_maj_std * weights).sum(axis=1)

        x = np.arange(n_pairs)
        axes[i].plot(x, avg_rdx_maj, 'o-', label='RDX cluster', color='tab:red')
        axes[i].plot(x, avg_rand_mean, 's--', label='Random baseline', color='tab:blue')
        axes[i].fill_between(x, avg_rand_mean - avg_rand_std, avg_rand_mean + avg_rand_std,
                             alpha=0.2, color='tab:blue')
        axes[i].axhline(overall_majority_frac, ls=':', color='tab:gray', label='Overall majority frac')
        axes[i].set_title(f'Majority-class fraction in clusters (dir={direction_str})')
        axes[i].set_xlabel('Layer Pair')
        axes[i].set_ylabel('Majority-class fraction')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
        axes[i].legend(fontsize=8)
    plt.tight_layout()
    finish_plot(show, save, os.path.join(fig_output_dir, 'class_majority_fraction.png') if fig_output_dir else None,
                fig=fig)

    # --- Plot 7: Class homogeneity enrichment (rdx - random) heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, direction_str in enumerate(['01', '10']):
        enrichment = class_homogeneity[i, :, 1:, 0] - class_homogeneity[i, :, 1:, 1]
        enrichment_masked = _mask_heatmap(enrichment, cluster_exists[i])
        vmax = np.nanmax(np.abs(enrichment_masked))
        im = axes[i].imshow(enrichment_masked.T, cmap='bwr', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[i].set_title(f'Class homogeneity enrichment (dir={direction_str})')
        axes[i].set_xlabel('Layer Pair')
        axes[i].set_ylabel('Cluster')
        axes[i].set_xticks(range(n_pairs))
        axes[i].set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
        axes[i].set_yticks(range(n_clusters))
        axes[i].set_yticklabels([f'C{c + 1}' for c in range(n_clusters)])
        fig.colorbar(im, ax=axes[i], label='Majority frac difference')
        for ki_idx in range(n_pairs):
            for ci in range(n_clusters):
                if cluster_exists[i, ki_idx, ci] and class_homogeneity[i, ki_idx, ci + 1, 3] < 0.05:
                    axes[i].text(ki_idx, ci, '*', ha='center', va='center', color='black', fontsize=12,
                                 fontweight='bold')
    plt.tight_layout()
    finish_plot(show, save,
                os.path.join(fig_output_dir, 'class_homogeneity_enrichment.png') if fig_output_dir else None, fig=fig)

    # --- Plot 8: Fraction of class-selective clusters per layer pair ---
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(n_pairs)
    width = 0.35
    for i, direction_str in enumerate(['01', '10']):
        pvals = class_homogeneity[i, :, 1:, 3]
        sig_count = np.array([(pvals[ki, cluster_exists[i, ki]] < 0.05).sum() for ki in range(n_pairs)])
        denom = np.maximum(n_actual_clusters[i], 1)
        frac_sig = sig_count / denom
        ax.bar(x + i * width - width / 2, frac_sig, width, label=f'dir={direction_str}')
    ax.set_xlabel('Layer Pair')
    ax.set_ylabel('Fraction of clusters with p < 0.05')
    ax.set_title('Fraction of class-selective clusters per layer pair')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.set_ylim([0, 1])
    plt.tight_layout()
    finish_plot(show, save,
                os.path.join(fig_output_dir, 'fraction_class_selective_clusters.png') if fig_output_dir else None,
                fig=fig)


def compute_activations(visual, replaced_attn, ds, inds, args, cache_path, force_run=False, collect_head_acts=True):
    # cache activations at target layers on subset for act patching experiments
    if not os.path.exists(cache_path) or force_run:

        dataloader = DataLoader(
            torch.utils.data.Subset(ds, inds),
            batch_size=args.batch_size, shuffle=False, num_workers=4,
        )
        # Collect attn head outputs via hooks: reshape (B, S, C) -> (B, S, H, D)
        # Only store running sum + count to compute mean, avoiding O(N*S*H*D) RAM
        attn_head_sums = {name: None for name, _ in replaced_attn}
        attn_head_counts = {name: 0 for name, _ in replaced_attn}
        if collect_head_acts:
            for name, m in replaced_attn:
                num_heads, head_dim = m.num_heads, m.head_dim

                def make_collect_hook(sums, counts, key, num_heads, head_dim):
                    def collect_hook(x):
                        B, S, C = x.shape
                        batch_sum = x.reshape(B, S, num_heads, head_dim).detach().cpu().sum(dim=0)
                        if sums[key] is None:
                            sums[key] = batch_sum
                        else:
                            sums[key] += batch_sum
                        counts[key] += B
                        return x

                    return collect_hook

                m.register_hook('attn_output',
                                make_collect_hook(attn_head_sums, attn_head_counts, name, num_heads, head_dim))

        # Collect block outputs: (N, S, D)
        block_names, block_modules = find_transformer_blocks(visual)
        block_hook = ActivationHookV2(move_to_cpu_in_hook=True)
        block_hook.register_hooks(block_names, block_modules)

        # Collect pre-block embedding (input to first transformer block)
        pre_block_name = 'pre_' + block_names[0]
        pre_block_acts = []

        def pre_block_collect_hook(module, input):
            x = input[0].detach().clone().cpu()
            pre_block_acts.append(x)

        pre_block_handle = block_modules[0].register_forward_pre_hook(pre_block_collect_hook)

        # Collect post-final-layer-norm activations
        post_ln_acts = []
        ln_module = (getattr(visual, 'ln_post', None) or getattr(visual, 'fc_norm', None) or getattr(visual, 'norm',
                                                                                                     None)
                     or getattr(getattr(visual, 'trunk', None), 'norm', None))

        def post_ln_collect_hook(module, input, output):
            post_ln_acts.append(output.detach().clone().cpu())

        post_ln_handle = ln_module.register_forward_hook(post_ln_collect_hook) if ln_module is not None else None

        # Collect post-projection (final 512-dim output)
        post_proj_acts = []

        # Forward pass
        all_labels = []
        visual.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Collecting activations for patching'):
                images = batch['input'].to(args.device)
                all_labels.append(batch['target'])
                out = visual(images)
                post_proj_acts.append(out.detach().clone().cpu())

        labels = torch.cat(all_labels, dim=0)

        if collect_head_acts:
            # Compute mean attn head outputs: (S, H, D) per layer
            attn_head_acts = {name: attn_head_sums[name] / attn_head_counts[name] for name in attn_head_sums if
                              attn_head_sums[name] is not None}
        else:
            attn_head_acts = {}

        # Concatenate block outputs
        block_hook.concatenate_layer_activations()
        pre_block_embedding = torch.cat(pre_block_acts, dim=0)
        block_acts = {pre_block_name: pre_block_embedding}
        block_acts.update({k: v for k, v in block_hook.layer_activations.items()})

        # Concatenate post-ln and post-proj activations
        post_ln_embedding = torch.cat(post_ln_acts, dim=0) if post_ln_acts else None
        post_proj_embedding = torch.cat(post_proj_acts, dim=0)

        activations = {
            'attn_head': attn_head_acts,  # {name: (S, H, D)} mean per-head attn output
            'block': block_acts,  # {name: (N, S, D)} block output
            'post_ln': post_ln_embedding,  # (N, D) or (N, S, D) after final layer norm
            'post_proj': post_proj_embedding,  # (N, 512) after final projection
        }

        # Cleanup
        block_hook.remove_hooks()
        pre_block_handle.remove()
        if post_ln_handle is not None:
            post_ln_handle.remove()
        for _, m in replaced_attn:
            m.clear_hooks()

        with open(cache_path, 'wb') as f:
            pkl.dump(dict(activations=activations, labels=labels), f)
    else:
        with open(cache_path, 'rb') as f:
            cached = pkl.load(f)
            activations = cached['activations']
            labels = cached['labels']

    return activations, labels


# ── Activation patching experiment ───────────────────────────────────────────
def analyze_biomedclip(visual, train_ds, args, force_run=False):
    """
    Collect Activations
    1) run forward pass on subset with collection hooks at
        1) attn head outputs N, H, S, D
        2) Block outputs N, S, D

    Patch one block at a time
    1) For each block, patch each attention head output with mean activations for all inputs

    Metric - at each block, measure the top 16 nearest neighbor change for each sample (N)
    """
    fig_output_dir = os.path.join(args.output_dir, 'fig_outputs')
    os.makedirs(fig_output_dir, exist_ok=True)

    from src.utils.hookable_timm_modules import replace_attention_modules
    replaced_attn = replace_attention_modules(visual)
    rng = np.random.default_rng(seed=0)
    inds = rng.choice(len(train_ds), args.num_samples + args.probe_num_samples, replace=False)
    ptch_inds = inds[:args.num_samples]
    probe_inds = inds[args.num_samples:]

    with open(os.path.join(args.output_dir, 'inds_for_patching_and_probing.pkl'), 'wb') as f:
        pkl.dump({'ptch_inds': ptch_inds, 'probe_inds': probe_inds}, f)

    print('Computing activations for patching and probing subsets...')
    cache_path = os.path.join(args.output_dir, 'full_cache_subset.pkl')
    activations, labels = compute_activations(visual, replaced_attn, train_ds, ptch_inds, args,
                                              cache_path, force_run=force_run, collect_head_acts=True)
    cache_path = os.path.join(args.output_dir, 'probe_subset.pkl')
    probe_activations, probe_labels = compute_activations(visual, replaced_attn, train_ds, probe_inds, args, cache_path,
                                                          force_run=force_run, collect_head_acts=False)

    print('Grabbing class token for activations...')
    acts = {}
    probe_acts = {}
    probe_results = {}
    for i, (k, v) in enumerate(activations['block'].items()):
        acts[k] = activations['block'][k][:, 0]
        probe_acts[k] = probe_activations['block'][k][:, 0]

    acts['post_ln'] = activations['post_ln'][:, 0] if activations['post_ln'] is not None else activations['post_proj']
    probe_acts['post_ln'] = (probe_activations['post_ln'][:, 0] if probe_activations['post_ln'] is not None
                             else probe_activations['post_proj'])
    acts['post_proj'] = activations['post_proj']
    probe_acts['post_proj'] = probe_activations['post_proj']

    probe_results_path = os.path.join(args.output_dir, 'probe_results.pkl')
    if not os.path.exists(probe_results_path) or force_run:
        probe_results['knn_train'] = run_knn_probes_full(probe_acts, probe_labels, args.topk)
        probe_results['knn_test'] = evaluate_probes(probe_results['knn_train'], acts, np.array(labels))
        probe_results['linear_train'] = run_probes_full(probe_acts, probe_labels)
        probe_results['linear_test'] = evaluate_probes(probe_results['linear_train'], acts, np.array(labels))
        with open(probe_results_path, 'wb') as f:
            pkl.dump(probe_results, f)
        with open(os.path.join(fig_output_dir, 'accuracies.txt'), 'w') as f:
            for probe_method in ['knn', 'linear']:
                f.write(f"=== {probe_method.upper()} Probes ===\n")
                for key in probe_results[f'{probe_method}_test']:
                    f.write(f"{key} : Train {probe_results[f'{probe_method}_train'][key]['train_acc']:.4f} | "
                            f"Test {probe_results[f'{probe_method}_test'][key]['accuracy']:.4f}\n")
                f.write('\n')
    else:
        with open(probe_results_path, 'rb') as f:
            probe_results = pkl.load(f)

    plot_probe_accuracies(probe_results, args, fig_output_dir=fig_output_dir, show=True, save=True)

    null_thresh = 1.1
    rdx_params = {
        "method": "rdx",
        "method_name": "rdx_nb_lb",
        "sim_function": "neighborhood",
        "diff_function": "locally_biased",
        "clustering_method": "spectral",
        "symmetrize_diff": False,
        "symmetrize_dm": "mean",
        "filter_thresh": null_thresh,
        "add_null_cluster": True,
        "gamma_scale": None,
        "gamma": 0.05,
        "beta": 5,
        "seed": 0,
        "guidance": None,
        "n_clusters": 10,
        "viz_params": {
            "show": False,
            "save": True,
            "null_thresh": null_thresh,
            "num_samples": 16,
            "grid_size": "4x4",
            "skip_low_affinity_for_summary": False,
            "cluster_sample_strategy": "maximize_total_neighborhood_affinity",
            "label_cluster_images": True,
            "add_predicted_label_to_cluster_images": True
        }
    }

    probe_method = 'knn'
    test_acc = []
    for key in probe_results[f'{probe_method}_test']:
        print(f"{key} : Train {probe_results[f'{probe_method}_train'][key]['train_acc']:.4f} | "
              f"Test {probe_results[f'{probe_method}_test'][key]['accuracy']:.4f}")
        test_acc.append(probe_results[f'{probe_method}_test'][key]['accuracy'])

    test_acc = np.array(test_acc)
    print('Test accuracy across layers:', test_acc)

    pred_labels = dict([(key, probe_results[f'{probe_method}_test'][key]['predictions']) for key in
                        probe_results[f'{probe_method}_test']])

    data = run_rdx(rdx_params, copy.deepcopy(train_ds), acts, labels, pred_labels,
                   ptch_inds, args, force_run=force_run)
    show = True
    save = True

    measure_cluster_properties(rdx_params, data, probe_results, labels, probe_method=probe_method,
                               fig_output_dir=fig_output_dir, show=show, save=save)

    block_names, block_modules = find_transformer_blocks(visual)
    pre_block_name = 'pre_' + block_names[0]
    all_layer_names = [pre_block_name] + block_names
    torch.cuda.empty_cache()

    output_path = os.path.join(args.output_dir, 'patching_results.pkl')
    if os.path.exists(output_path) and not force_run:
        with open(output_path, 'rb') as f:
            cached = pkl.load(f)
            metrics = cached['metrics']
            probe_predictions = cached['probe_predictions']
    else:
        metrics, probe_predictions = activation_patching(replaced_attn, block_modules, all_layer_names, activations,
                                                         labels, data, probe_results, rdx_params, output_path, args.topk)

    knn_base_scores = np.array([probe_results['knn_test'][key]['accuracy'] for key in probe_results['knn_test']])[
        1:-2, None]
    linear_base_scores = np.array(
        [probe_results['linear_test'][key]['accuracy'] for key in probe_results['linear_test']])[1:-2, None]
    metric_names = ['CKA', '1 - TopKNS', 'RDX_NBDD', 'RDX_NBDD_SS', 'RDX_TopK_NBDD', 'RDX_TopK_NBDD_SS',
                    'KNN Acc', 'Linear Acc']
    for mi, metric in enumerate(metrics):
        block_head_score = np.array(metrics[metric])
        block_head_score = block_head_score - knn_base_scores if 'knn_acc' == metric \
            else block_head_score - linear_base_scores if 'linear_acc' == metric else block_head_score
        if len(block_head_score.shape) == 3:
            block_head_score = np.abs(block_head_score).sum(axis=2)
        if 'acc' in metric:
            norm = colors.CenteredNorm(vcenter=0)
            cmap = 'bwr'
        else:
            norm = None
            cmap = 'viridis'
        plt.figure()
        plt.imshow(block_head_score.T, cmap=cmap, norm=norm)
        plt.colorbar(label=metric_names[mi])
        plt.ylabel('Attention Head Index')
        plt.xlabel('Block Index')
        plt.xticks(ticks=range(len(all_layer_names) - 1), labels=[bn.split(".")[-1] for bn in all_layer_names[1:]])
        plt.title(f'Patching {metric} Score')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        finish_plot(show, save, os.path.join(fig_output_dir, f'patching_{metric}.png'))

    for metric in ['rdx_clnb_dist_delta', 'rdx_clnb_dist_delta_selectivity']:
        scores = np.array(metrics[metric])
        if len(scores.shape) < 3:
            continue
        # Skip per-direction/per-cluster plots for per-sample outputs (too large)
        num_entries_per_direction = scores.shape[-1] // 2
        max_cluster_entries = rdx_params['n_clusters'] + 2 if rdx_params.get('n_clusters') is not None else 50
        if num_entries_per_direction > max_cluster_entries:
            continue
        mdpt = scores.shape[-1] // 2
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, direct in enumerate(['01', '10']):
            direct_scores = scores[:, :, :mdpt] if direct == '01' else scores[:, :, mdpt:]
            im = axes[i].imshow(direct_scores.mean(axis=2).T, cmap='viridis')
            axes[i].set_title(f'RDX NB Distance Delta {direct}')
            axes[i].set_xlabel('Block Index')
            axes[i].set_ylabel('Attention Head Index')
            axes[i].set_xticks(ticks=range(len(all_layer_names) - 1),
                               labels=[bn.split(".")[-1] for bn in all_layer_names[1:]])
            axes[i].invert_yaxis()
            fig.colorbar(im, ax=axes[i], label='Relative Neighborhood Distance Delta')
        plt.tight_layout()
        finish_plot(show, save, os.path.join(fig_output_dir, f'patching_{metric}_by_direction.png'), fig=fig)

    num_clusters = len(metrics['rdx_clnb_dist_delta'][0][0]) // 2
    for metric in ['rdx_clnb_dist_delta', 'rdx_clnb_dist_delta_selectivity']:
        scores = np.array(metrics[metric])
        if len(scores.shape) < 3:
            continue
        # Skip per-cluster plots for per-sample outputs (too large)
        num_entries_per_direction = scores.shape[-1] // 2
        max_cluster_entries = rdx_params['n_clusters'] + 2 if rdx_params.get('n_clusters') is not None else 50
        if num_entries_per_direction > max_cluster_entries:
            continue
        mdpt = scores.shape[-1] // 2
        fig, axes = plt.subplots(num_clusters, 2, figsize=(10, 4 * num_clusters))
        axes = axes.T
        for i, direct in enumerate(['01', '10']):
            direct_scores = scores[:, :, :mdpt] if direct == '01' else scores[:, :, mdpt:]
            for di in range(direct_scores.shape[2]):
                im = axes[i, di].imshow(direct_scores[:, :, di].T, cmap='viridis')
                axes[i, di].set_title(f'{direct} Cluster {di + 1}')
                axes[i, di].set_xlabel('Block Index')
                axes[i, di].set_ylabel('Attention Head Index')
                axes[i, di].set_xticks(ticks=range(len(all_layer_names) - 1),
                                       labels=[bn.split(".")[-1] for bn in all_layer_names[1:]])
                axes[i, di].invert_yaxis()
                fig.colorbar(im, ax=axes[i, di])
        plt.tight_layout()
        finish_plot(show, save, os.path.join(fig_output_dir, f'patching_{metric}_per_cluster.png'), fig=fig)

    tmp = []
    for layer_pair in data['output_dict']:
        for direction in ['10']:
            d = data['output_dict'][layer_pair]['graph_dict'][f'am_{direction}'].topk(args.topk).values.mean(-1)
            tmp.append(d.cpu().numpy())
    aff_all = np.stack(tmp)
    sorted_inds = np.argsort(aff_all.mean(0))

    fontsize = 24
    for metric in ['rdx_topk_nb_dist_delta', 'rdx_topk_nb_dist_delta_selectivity']:
        scores = np.array(metrics[metric])
        mdpt = scores.shape[-1] // 2
        fig, axes = plt.subplots(2, 1, figsize=(30, 15), squeeze=False)
        num_samples = mdpt
        for i, direct in enumerate(['01', '10']):
            direct_scores = scores[:, :, :mdpt] if direct == '01' else scores[:, :, mdpt:]
            direct_scores = direct_scores[:, :, sorted_inds]
            di = 0
            im = axes[i, di].imshow(direct_scores.reshape(-1, num_samples), cmap='viridis')
            # axes[i, di].set_title(f'{direct} Cluster {di + 1}')
            axes[i, di].set_ylabel('Layer-Block Index', fontsize=fontsize)
            axes[i, di].set_xlabel(f'RDX Affinity ({direct}) Image Neighborhood', fontsize=fontsize)
            # axes[i, di].set_xticks(ticks=range(len(all_layer_names) - 1), labels=[bn.split(".")[-1] for bn in all_layer_names[1:]])
            axes[i, di].invert_yaxis()
            fig.colorbar(im, ax=axes[i, di])
        plt.tight_layout()
        finish_plot(show, save, os.path.join(fig_output_dir, f'patching_{metric}.png'), fig=fig)

    subsets = {
        'random': np.random.choice(sorted_inds, 9, replace=False),
        'high_avg_affinity': sorted_inds[-9:],
        'low_avg_affinity': sorted_inds[:9],
        'manual': [440, 113, 491,  # 10 -> 11
                   212,  # 1->2,
                   427, 330,  # 8-> 9,
                   345,  # 0-> 1,
                   419, 218  # pre -> 0
                   ]
    }
    for title, inds in subsets.items():
        for metric in ['rdx_topk_nb_dist_delta', 'rdx_topk_nb_dist_delta_selectivity']:
            scores = np.array(metrics[metric])
            mdpt = scores.shape[-1] // 2
            for i, direct in enumerate(['01', '10']):
                fig, axes = plt.subplots(3, 3, figsize=(18, 15))
                axes = axes.flatten()
                direct_scores = scores[:, :, :mdpt] if direct == '01' else scores[:, :, mdpt:]
                for ii in range(len(inds)):
                    ax = axes[ii]
                    im = ax.imshow(direct_scores[:, :, inds[ii]].T, cmap='viridis')
                    ax.set_title('Image Index: ' + str(inds[ii]), fontsize=fontsize)
                    ax.set_xlabel('Block Index', fontsize=fontsize)
                    ax.set_ylabel('Attention Head Index', fontsize=fontsize)
                    ax.set_xticks(ticks=range(len(all_layer_names) - 1),
                                  labels=[bn.split(".")[-1] for bn in all_layer_names[1:]])
                    ax.invert_yaxis()
                    fig.colorbar(im, ax=ax)
                plt.suptitle(f'{metric} - {direct}', fontsize=fontsize)
                plt.tight_layout()
                finish_plot(show, save, os.path.join(fig_output_dir, f'patching_{metric}_{direct}_{title}_subset.png'),
                            fig=fig)


# ── RDX experiment ───────────────────────────────────────────────────────────
def run_rdx(rdx_params, train_ds, acts, labels, pred_labels, inds, args, force_run=False):
    # Remove normalize for plotting raw images in RDX viz
    train_ds.transform = torchvision.transforms.Compose(train_ds.transform.transforms[:-1])
    image_samples = [train_ds[i]['input'] for i in tqdm(inds)]
    image_samples = torch.stack(image_samples)

    rdx_params['image_samples'] = image_samples
    rdx_params['dataset_labels'] = labels

    layer_names = list(acts.keys())
    exp_pairs = list(zip(layer_names[:-1], layer_names[1:]))

    data_dict = {'fig_data': {}, 'output_dict': {}}
    for ln1, ln2 in exp_pairs:
        rdx_params['representations'] = [torch.tensor(acts[ln1]), torch.tensor(acts[ln2])]
        rdx_params['preds'] = np.stack([pred_labels[ln1], pred_labels[ln2]])
        print(f"Running RDX for {ln1} vs. {ln2} ...")
        method_dir = os.path.join(args.output_dir, 'rdx_outputs', f"rdx_{ln1}_vs_{ln2}")
        os.makedirs(method_dir, exist_ok=True)
        rdx = RDX()

        if os.path.exists(os.path.join(method_dir, 'outputs.pkl')) and not force_run:
            with open(os.path.join(method_dir, 'outputs.pkl'), 'rb') as f:
                output_dict = pkl.load(f)

            if rdx_params.get('viz_params', None) is not None:
                with open(os.path.join(method_dir, 'fig_data.pkl'), 'rb') as f:
                    fig_data = pkl.load(f)
        else:

            output_dict = rdx.fit(rdx_params)
            output_dict['method_dir'] = method_dir
            with open(os.path.join(method_dir, 'outputs.pkl'), 'wb') as f:
                pkl.dump(output_dict, f)

            if rdx_params.get('viz_params', None) is not None:
                tsne0 = TSNE(2, perplexity=16, early_exaggeration=12, verbose=1)
                tsne1 = TSNE(2, perplexity=16, early_exaggeration=12, verbose=1)
                rdx_params['red0'] = tsne0.fit_transform(rdx_params['representations'][0])
                rdx_params['red1'] = tsne1.fit_transform(rdx_params['representations'][1])

                fig_data = rdx.generate_visualizations(rdx_params, output_dict, rdx_params['viz_params'])
                with open(os.path.join(method_dir, 'fig_data.pkl'), 'wb') as f:
                    pkl.dump(fig_data, f)

        data_dict['output_dict'][(ln1, ln2)] = output_dict
        data_dict['fig_data'][(ln1, ln2)] = fig_data

    return data_dict


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = parse_args(description=__doc__)
    parser.add_argument('--model_name', type=str, default='biomedclip', help='Name of the model to use')
    parser.add_argument('--num_samples', type=int, default=2500,
                        help='Number of samples to use for patching experiments')
    parser.add_argument('--probe_num_samples', type=int, default=3000, )
    parser.add_argument('--force_run', action='store_true',
                        help='Force re-computation of cached activations')
    parser.add_argument('--topk', type=int, default=16, help='Top K for neighborhood-based metrics')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'../outputs/rsna_{args.model_name}/'
    os.makedirs(args.output_dir, exist_ok=True)

    visual, preprocess = load_model(args.model_name, args.device)
    train_ds = load_dataset(args.data_root, preprocess)
    analyze_biomedclip(visual, train_ds, args, force_run=args.force_run)


if __name__ == '__main__':
    main()
