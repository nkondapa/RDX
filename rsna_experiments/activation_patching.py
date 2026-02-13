"""
Activation patching + RDX experiments on BiomedCLIP (RSNA Pneumonia).

Usage:
    python -m rsna_experiments.activation_patching [--data_root PATH] [--output_dir PATH]
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from matplotlib import colors

from src.utils.hooks import ActivationHookV2
from src.rdx import RDX
from rsna_experiments.utils import (
    parse_args, load_model, load_dataset,
    cls_token, find_transformer_blocks, cache_activations,
)
from src.cka import CKA
from src.utils.plotting_helper import finish_plot
from linear_probes import run_probes, run_knn_probes, run_probes_full, run_knn_probes_full, evaluate_probes


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


def relative_neighborhood_distance_delta(base_embs, patched_embs, target_embs, rdx_data, rdx_fig_data, k=16):
    # base_dist = torch.cdist(base_embs, base_embs)
    patched_dist = torch.cdist(patched_embs, patched_embs)
    # target_dist = torch.cdist(target_embs, target_embs)
    base_dist = rdx_data['graph_dict']['r0_dm']
    patched_dist = patched_dist.argsort(-1).argsort(-1).type(torch.float32)
    target_dist = rdx_data['graph_dict']['r1_dm']
    # dist1 = base_dist.argsort(-1).argsort(-1)
    # dist2 = target_dist.argsort(-1).argsort(-1)
    # for di in ['01', '10']:
    #     for sel_inds in rdx_fig_data[di]['selected_indices']:
    #         print(sel_inds)
    #         if di == '01':
    #             print(dist1[sel_inds[0], sel_inds])
    #         else:
    #             print(dist2[sel_inds[0], sel_inds])
    #     print()

    scores = []
    selectivity_scores = []
    for direct in ['01', '10']:
        # sel_inds_per_cluster = rdx_fig_data[direct]['selected_indices']
        array = rdx_data['cluster_dict'][f'{direct}']['cluster_labels']
        sel_inds_per_cluster = [((array == i).nonzero()[0], (array != i).nonzero()[0]) for i in np.unique(array)]
        for i, (sel_inds, not_sel_inds) in enumerate(sel_inds_per_cluster[1:]):
            patched_sel_dist = patched_dist[sel_inds][:, sel_inds].mean()
            target_sel_dist = target_dist[sel_inds][:, sel_inds].mean()
            base_sel_dist = base_dist[sel_inds][:, sel_inds].mean()
            patched_nsel_dist = patched_dist[not_sel_inds][:, not_sel_inds]
            target_nsel_dist = target_dist[not_sel_inds][:, not_sel_inds]
            non_cluster_change = torch.abs(target_nsel_dist - patched_nsel_dist).mean() + 1e-8
            if direct == '01':
                # expectation : target dist is larger than base dist
                # positive if patch made samples closer together, means head is more important
                rdx_nb_dd = -1 * (patched_sel_dist - target_sel_dist)
            else:
                # expectation : target dist is smaller than base dist
                # positive if patch made samples further apart, means head is more important
                rdx_nb_dd = (patched_sel_dist - target_sel_dist)
            selectivity_score = rdx_nb_dd / non_cluster_change
            selectivity_scores.append(selectivity_score.item())
            scores.append(rdx_nb_dd.item())

            # print(direct, i, patched_sel_dist.item(), target_sel_dist.item(), base_sel_dist.item(), rdx_nb_dd)

    return scores, selectivity_scores

def measure_cluster_properties(rdx_params, data, probe_results, labels, probe_method='knn',
                               fig_output_dir=None, show=True, save=False):

    num_perms = 1000
    n_clusters = rdx_params['n_clusters']
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
                p_value = (random_rates[:, labi] >= rdx_rate).sum() / num_perms
                cluster_enrichment[i, ki, labi] = [rdx_rate, random_rates[:, labi].mean(),
                                                   random_rates[:, labi].std(), p_value]
                print(f'  {key} dir={direction_str} cluster={labi} | rdx_rate={rdx_rate:.4f} '
                      f'random={random_rates[:, labi].mean():.4f}+/-{random_rates[:, labi].std():.4f} '
                      f'p={p_value:.4f} (n={size})')

    pred_change_freq = np.array(pred_change_freq)
    layer_pair_keys = list(data['output_dict'].keys())
    layer_pair_labels = [f'{k[0].split(".")[-1]}-{k[1].split(".")[-1]}' for k in layer_pair_keys]
    n_pairs = len(layer_pair_keys)

    # --- Plot 1: p-value heatmap (directions x clusters) across layer pairs ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, direction_str in enumerate(['01', '10']):
        # cluster_enrichment[i, :, :, 3] is the p-value; skip null cluster (index 0)
        pvals = cluster_enrichment[i, :, 1:, 3]  # (n_pairs, n_clusters)
        im = axes[i].imshow(pvals.T, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[i].set_title(f'Enrichment p-value (dir={direction_str})')
        axes[i].set_xlabel('Layer Pair')
        axes[i].set_ylabel('Cluster')
        axes[i].set_xticks(range(n_pairs))
        axes[i].set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
        axes[i].set_yticks(range(n_clusters))
        axes[i].set_yticklabels([f'C{c+1}' for c in range(n_clusters)])
        fig.colorbar(im, ax=axes[i], label='p-value')
        # Mark significant cells
        for ki_idx in range(n_pairs):
            for ci in range(n_clusters):
                if pvals[ki_idx, ci] < 0.05:
                    axes[i].text(ki_idx, ci, '*', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    plt.tight_layout()
    finish_plot(show, save, os.path.join(fig_output_dir, 'cluster_enrichment_pvalues.png') if fig_output_dir else None, fig=fig)

    # --- Plot 2: RDX rate vs random baseline per layer pair (non-null clusters aggregated) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, direction_str in enumerate(['01', '10']):
        rdx_rates = cluster_enrichment[i, :, 1:, 0]      # (n_pairs, n_clusters)
        rand_means = cluster_enrichment[i, :, 1:, 1]
        rand_stds = cluster_enrichment[i, :, 1:, 2]
        cluster_sizes = nc[i, :, 1:, 1]                   # (n_pairs, n_clusters)

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
    finish_plot(show, save, os.path.join(fig_output_dir, 'cluster_pred_change_rate.png') if fig_output_dir else None, fig=fig)

    # --- Plot 3: Per-cluster enrichment (rdx_rate - random_mean) heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, direction_str in enumerate(['01', '10']):
        enrichment = cluster_enrichment[i, :, 1:, 0] - cluster_enrichment[i, :, 1:, 1]  # (n_pairs, n_clusters)
        vmax = np.abs(enrichment).max()
        im = axes[i].imshow(enrichment.T, cmap='bwr', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[i].set_title(f'Enrichment (rdx - random) (dir={direction_str})')
        axes[i].set_xlabel('Layer Pair')
        axes[i].set_ylabel('Cluster')
        axes[i].set_xticks(range(n_pairs))
        axes[i].set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
        axes[i].set_yticks(range(n_clusters))
        axes[i].set_yticklabels([f'C{c+1}' for c in range(n_clusters)])
        fig.colorbar(im, ax=axes[i], label='Rate difference')
        # Mark significant cells
        for ki_idx in range(n_pairs):
            for ci in range(n_clusters):
                if cluster_enrichment[i, ki_idx, ci + 1, 3] < 0.05:
                    axes[i].text(ki_idx, ci, '*', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    plt.tight_layout()
    finish_plot(show, save, os.path.join(fig_output_dir, 'cluster_enrichment_diff.png') if fig_output_dir else None, fig=fig)

    # --- Plot 4: Fraction of significant clusters per layer pair ---
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(n_pairs)
    width = 0.35
    for i, direction_str in enumerate(['01', '10']):
        pvals = cluster_enrichment[i, :, 1:, 3]  # (n_pairs, n_clusters)
        frac_sig = (pvals < 0.05).sum(axis=1) / n_clusters
        ax.bar(x + i * width - width / 2, frac_sig, width, label=f'dir={direction_str}')
    ax.set_xlabel('Layer Pair')
    ax.set_ylabel('Fraction of clusters with p < 0.05')
    ax.set_title('Fraction of significantly enriched clusters per layer pair')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.set_ylim([0, 1])
    plt.tight_layout()
    finish_plot(show, save, os.path.join(fig_output_dir, 'fraction_significant_clusters.png') if fig_output_dir else None, fig=fig)

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
                p_value = (random_maj_frac[:, labi] >= rdx_maj_frac).sum() / num_perms
                class_homogeneity[i, ki, labi] = [rdx_maj_frac, random_maj_frac[:, labi].mean(),
                                                   random_maj_frac[:, labi].std(), p_value]
                print(f'  {key} dir={direction_str} cluster={labi} | maj_frac={rdx_maj_frac:.4f} '
                      f'random={random_maj_frac[:, labi].mean():.4f}+/-{random_maj_frac[:, labi].std():.4f} '
                      f'p={p_value:.4f} (n={size})')

    # --- Plot 5: Class homogeneity p-value heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, direction_str in enumerate(['01', '10']):
        pvals = class_homogeneity[i, :, 1:, 3]  # (n_pairs, n_clusters)
        im = axes[i].imshow(pvals.T, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[i].set_title(f'Class homogeneity p-value (dir={direction_str})')
        axes[i].set_xlabel('Layer Pair')
        axes[i].set_ylabel('Cluster')
        axes[i].set_xticks(range(n_pairs))
        axes[i].set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
        axes[i].set_yticks(range(n_clusters))
        axes[i].set_yticklabels([f'C{c+1}' for c in range(n_clusters)])
        fig.colorbar(im, ax=axes[i], label='p-value')
        for ki_idx in range(n_pairs):
            for ci in range(n_clusters):
                if pvals[ki_idx, ci] < 0.05:
                    axes[i].text(ki_idx, ci, '*', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    plt.tight_layout()
    finish_plot(show, save, os.path.join(fig_output_dir, 'class_homogeneity_pvalues.png') if fig_output_dir else None, fig=fig)

    # --- Plot 6: Majority-class fraction: RDX vs random baseline ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, direction_str in enumerate(['01', '10']):
        rdx_maj = class_homogeneity[i, :, 1:, 0]       # (n_pairs, n_clusters)
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
    finish_plot(show, save, os.path.join(fig_output_dir, 'class_majority_fraction.png') if fig_output_dir else None, fig=fig)

    # --- Plot 7: Class homogeneity enrichment (rdx - random) heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, direction_str in enumerate(['01', '10']):
        enrichment = class_homogeneity[i, :, 1:, 0] - class_homogeneity[i, :, 1:, 1]
        vmax = np.abs(enrichment).max()
        im = axes[i].imshow(enrichment.T, cmap='bwr', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[i].set_title(f'Class homogeneity enrichment (dir={direction_str})')
        axes[i].set_xlabel('Layer Pair')
        axes[i].set_ylabel('Cluster')
        axes[i].set_xticks(range(n_pairs))
        axes[i].set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
        axes[i].set_yticks(range(n_clusters))
        axes[i].set_yticklabels([f'C{c+1}' for c in range(n_clusters)])
        fig.colorbar(im, ax=axes[i], label='Majority frac difference')
        for ki_idx in range(n_pairs):
            for ci in range(n_clusters):
                if class_homogeneity[i, ki_idx, ci + 1, 3] < 0.05:
                    axes[i].text(ki_idx, ci, '*', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    plt.tight_layout()
    finish_plot(show, save, os.path.join(fig_output_dir, 'class_homogeneity_enrichment.png') if fig_output_dir else None, fig=fig)

    # --- Plot 8: Fraction of class-selective clusters per layer pair ---
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(n_pairs)
    width = 0.35
    for i, direction_str in enumerate(['01', '10']):
        pvals = class_homogeneity[i, :, 1:, 3]
        frac_sig = (pvals < 0.05).sum(axis=1) / n_clusters
        ax.bar(x + i * width - width / 2, frac_sig, width, label=f'dir={direction_str}')
    ax.set_xlabel('Layer Pair')
    ax.set_ylabel('Fraction of clusters with p < 0.05')
    ax.set_title('Fraction of class-selective clusters per layer pair')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_pair_labels, rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.set_ylim([0, 1])
    plt.tight_layout()
    finish_plot(show, save, os.path.join(fig_output_dir, 'fraction_class_selective_clusters.png') if fig_output_dir else None, fig=fig)

def compute_activations(visual, replaced_attn, ds, inds, args, cache_path, force_run=False, collect_head_acts=True):

    # cache activations at target layers on subset for act patching experiments
    if not os.path.exists(cache_path) or force_run:

        dataloader = DataLoader(
            torch.utils.data.Subset(ds, inds),
            batch_size=args.batch_size, shuffle=False, num_workers=4,
        )
        # Collect attn head outputs via hooks: reshape (B, S, C) -> (B, S, H, D) and accumulate
        attn_head_acts = {name: [] for name, _ in replaced_attn}
        if collect_head_acts:
            for name, m in replaced_attn:
                num_heads, head_dim = m.num_heads, m.head_dim
                collector = attn_head_acts[name]

                def make_collect_hook(collector, num_heads, head_dim):
                    def collect_hook(x):
                        B, S, C = x.shape
                        collector.append(x.reshape(B, S, num_heads, head_dim).detach().cpu().clone())
                        return x

                    return collect_hook

                m.register_hook('attn_output', make_collect_hook(collector, num_heads, head_dim))

        # Collect block outputs: (N, S, D)
        block_names, block_modules = find_transformer_blocks(visual)
        block_hook = ActivationHookV2(move_to_cpu_in_hook=True)
        block_hook.register_hooks(block_names, block_modules)

        # Collect pre-block embedding (input to first transformer block)
        pre_block_name = 'trunk.embed'
        pre_block_acts = []
        def pre_block_collect_hook(module, input):
            x = input[0].detach().clone().cpu()
            pre_block_acts.append(x)
        pre_block_handle = block_modules[0].register_forward_pre_hook(pre_block_collect_hook)

        # Forward pass
        all_labels = []
        visual.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Collecting activations for patching'):
                images = batch['input'].to(args.device)
                all_labels.append(batch['target'])
                visual(images)

        labels = torch.cat(all_labels, dim=0)

        if collect_head_acts:
            # Concatenate attn head outputs
            attn_head_acts = {name: torch.cat(chunks, dim=0) for name, chunks in attn_head_acts.items()}

        # Concatenate block outputs
        block_hook.concatenate_layer_activations()
        pre_block_embedding = torch.cat(pre_block_acts, dim=0)
        block_acts = {pre_block_name: pre_block_embedding}
        block_acts.update({k: v for k, v in block_hook.layer_activations.items()})

        activations = {
            'attn_head': attn_head_acts,  # {name: (N, S, H, D)} per-head attn output
            'block': block_acts,  # {name: (N, S, D)} block output
        }

        # Cleanup
        block_hook.remove_hooks()
        pre_block_handle.remove()
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
def activation_patching(visual, train_ds, args, force_run=False):


    '''
    Collect Activations
    1) run forward pass on subset with collection hooks at
        1) attn head outputs N, H, S, D
        3) Block outputs N, S, D

    Patch one block at a time
    1) For each block, patch each attention head output with mean activations for all inputs

    Metric - at each block, measure the top 16 nearest neighbor change for each sample (N)
    '''

    from src.utils.hookable_timm_modules import replace_attention_modules, replace_mlp_modules
    replaced_attn = replace_attention_modules(visual)
    rng = np.random.default_rng(seed=0)
    inds = rng.choice(len(train_ds), args.num_samples + args.probe_num_samples, replace=False)
    ptch_inds = inds[:args.num_samples]
    probe_inds = inds[args.num_samples:]

    print('Computing activations for patching and probing subsets...')
    cache_path = os.path.join(args.output_dir, 'full_cache_subset.pkl')
    activations, labels = compute_activations(visual, replaced_attn, train_ds, ptch_inds, args, cache_path,
                                              force_run=force_run, collect_head_acts=True)
    cache_path = os.path.join(args.output_dir, 'probe_subset.pkl')
    probe_activations, probe_labels = compute_activations(visual, replaced_attn, train_ds, probe_inds, args, cache_path,
                                                          force_run=force_run, collect_head_acts=False)

    print('Grabbing class token for activations...')
    acts = {}
    probe_acts = {}
    probe_results = {}
    for i, (k, v) in enumerate(activations['block'].items()):
        acts[k] = activations['block'][k][:, -1]
        probe_acts[k] = probe_activations['block'][k][:, -1]

    probe_results_path = os.path.join(args.output_dir, 'probe_results.pkl')
    if not os.path.exists(probe_results_path) or force_run:
        probe_results['knn_train'] = run_knn_probes_full(probe_acts, probe_labels, 5)
        probe_results['knn_test'] = evaluate_probes(probe_results['knn_train'], acts, np.array(labels))
        probe_results['linear_train'] = run_probes_full(probe_acts, probe_labels)
        probe_results['linear_test'] = evaluate_probes(probe_results['linear_train'], acts, np.array(labels))
        with open(probe_results_path, 'wb') as f:
            pkl.dump(probe_results, f)
    else:
        with open(probe_results_path, 'rb') as f:
            probe_results = pkl.load(f)

    # probe_a = run_probes_full({'trunk.blocks.0': probe_acts['trunk.blocks.0']}, probe_labels, seed=42)
    # probe_b = run_probes_full({'trunk.blocks.0': probe_acts['trunk.blocks.0']}, probe_labels, seed=123)
    # probe_c = run_probes_full({'trunk.blocks.1': probe_acts['trunk.blocks.1']}, probe_labels, seed=123)
    #
    # same_layer_disagreement = (probe_a['trunk.blocks.0']['classifier'].predict(acts['trunk.blocks.0']) != probe_b['trunk.blocks.0']['classifier'].predict(acts['trunk.blocks.0'])).mean()
    # cross_layer_disagreement = (probe_a['trunk.blocks.0']['classifier'].predict(acts['trunk.blocks.0']) != probe_c['trunk.blocks.1']['classifier'].predict(acts['trunk.blocks.1'])).mean()
    # print(f"Same layer disagreement: {same_layer_disagreement:.3f}, Cross layer disagreement: {cross_layer_disagreement:.3f}")

    rdx_params = {
        "method": "rdx",
        "method_name": "rdx_nb_lb",
        "sim_function": "neighborhood",
        "diff_function": "locally_biased",
        "clustering_method": "spectral",
        "add_null_cluster": True,
        "gamma_scale": None,
        "gamma": 0.05,
        "beta": 5,
        "seed": 0,
        "guidance": None,
        "n_clusters": 4,
        "viz_params": {
            "show": False,
            "save": True,
            "null_thresh": 1.5,
            "num_samples": 16,
            "grid_size": "4x4",
            "skip_low_affinity_for_summary": False,
            # note this is a different selection strat than main paper
            # "cluster_sample_strategy": "maximize_total_euc_neighborhood_affinity"
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
    pred_labels = dict([(key, probe_results[f'{probe_method}_test'][key]['predictions']) for key in probe_results[f'{probe_method}_test']])

    data = run_rdx(rdx_params, copy.deepcopy(train_ds), acts, labels, pred_labels,
                   ptch_inds, args, force_run=False)

    show = True
    save = True
    fig_output_dir = os.path.join(args.output_dir, 'fig_outputs')
    measure_cluster_properties(rdx_params, data, probe_results, labels, probe_method=probe_method,
                               fig_output_dir=fig_output_dir, show=show, save=save)

    dist1 = torch.cdist(acts['trunk.blocks.0'], acts['trunk.blocks.0']).argsort(-1).argsort(-1)
    dist2 = torch.cdist(acts['trunk.blocks.1'], acts['trunk.blocks.1']).argsort(-1).argsort(-1)
    for sel_inds in data['fig_data'][('trunk.blocks.0', 'trunk.blocks.1')]['10']['selected_indices']:
        print(sel_inds)
        print(dist2[sel_inds[0], sel_inds])

    # d[('trunk.blocks.0', 'trunk.blocks.1')]['01']['selected_indices']
    block_names, block_modules = find_transformer_blocks(visual)
    pre_block_name = 'trunk.embed'
    all_layer_names = [pre_block_name] + block_names
    torch.cuda.empty_cache()

    ra_dict = dict(replaced_attn)
    batch_size = 256
    metrics = {'cka': [], 'topkns': [], 'rdx_nb_dist_delta': [], 'rdx_nb_dist_delta_selectivity': [],
               'knn_acc': [], 'linear_acc': []}
    probe_predictions = {'knn': [], 'linear': []}
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
            patch_val = activations['attn_head'][name][:, :, head_index, :].mean(dim=0)
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
            score = 1 - topk_neighbor_similarity(patched_acts[:, -1], activations['block'][target_layer][:, -1],
                                                 k=16)
            cka_score = 1 - cka_similarity(patched_acts[:, -1], activations['block'][target_layer][:, -1])
            if bi == 0 and head_index == 10:
                print()
            rdx_nb_dd, rndd_ss = relative_neighborhood_distance_delta(activations['block'][input_layer][:, -1],
                                                             patched_acts[:, -1],
                                                             activations['block'][target_layer][:, -1],
                                                             data['output_dict'][(input_layer, target_layer)],
                                                             data['fig_data'][(input_layer, target_layer)], k=16)
            act_dict = {target_layer: patched_acts[:, -1]}
            knn_result = evaluate_probes(probe_results['knn_train'], act_dict, np.array(labels), verbose=False)
            linear_result = evaluate_probes(probe_results['linear_train'], act_dict, np.array(labels), verbose=False)

            metrics['knn_acc'][-1].append(knn_result[target_layer]['accuracy'])
            metrics['linear_acc'][-1].append(knn_result[target_layer]['accuracy'])
            probe_predictions['knn'][-1].append(knn_result[target_layer]['predictions'])
            probe_predictions['linear'][-1].append(linear_result[target_layer]['predictions'])
            metrics['topkns'][-1].append(score)
            metrics['cka'][-1].append(cka_score)
            metrics['rdx_nb_dist_delta'][-1].append(rdx_nb_dd)
            metrics['rdx_nb_dist_delta_selectivity'][-1].append(rndd_ss)

            print(
                f'Block {target_layer} head {head_index} | topkns score: {score:.4f} | cka score: {cka_score:.4f} '
                f'| rdxnbdd score {np.abs(rdx_nb_dd).sum():.4f} | rdxnbddss score {np.abs(rndd_ss).sum():.4f}')
            m.clear_hooks()

    show = True
    save = True
    knn_base_scores = np.array([probe_results['knn_test'][key]['accuracy'] for key in probe_results['knn_test']])[1:, None]
    linear_base_scores = np.array([probe_results['linear_test'][key]['accuracy'] for key in probe_results['linear_test']])[1:, None]
    labels = ['CKA', '1 - TopKNS', 'RDX_NBDD', 'RDX_NBDD_SS', 'KNN Acc', 'Linear Acc']
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
        plt.colorbar(label=labels[mi])
        plt.ylabel('Attention Head Index')
        plt.xlabel('Block Index')
        plt.xticks(ticks=range(len(all_layer_names) - 1), labels=[bn.split(".")[-1] for bn in all_layer_names[1:]])
        plt.title(f'Patching {metric} Score')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        finish_plot(show, save, os.path.join(fig_output_dir, f'patching_{metric}.png'))

    for metric in ['rdx_nb_dist_delta', 'rdx_nb_dist_delta_selectivity']:
        scores = np.array(metrics[metric])
        mdpt = scores.shape[-1] // 2
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, direct in enumerate(['01', '10']):
            direct_scores = scores[:, :, :mdpt] if direct == '01' else scores[:, :, mdpt:]
            im = axes[i].imshow(direct_scores.mean(axis=2).T, cmap='viridis')
            axes[i].set_title(f'RDX NB Distance Delta {direct}')
            axes[i].set_xlabel('Block Index')
            axes[i].set_ylabel('Attention Head Index')
            axes[i].set_xticks(ticks=range(len(all_layer_names) - 1), labels=[bn.split(".")[-1] for bn in all_layer_names[1:]])
            axes[i].invert_yaxis()
            fig.colorbar(im, ax=axes[i], label='Relative Neighborhood Distance Delta')
        plt.tight_layout()
        finish_plot(show, save, os.path.join(fig_output_dir, f'patching_{metric}_by_direction.png'), fig=fig)

    num_clusters = len(metrics['rdx_nb_dist_delta'][0][0]) // 2
    for metric in ['rdx_nb_dist_delta', 'rdx_nb_dist_delta_selectivity']:
        scores = np.array(metrics[metric])
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
                axes[i, di].set_xticks(ticks=range(len(all_layer_names) - 1), labels=[bn.split(".")[-1] for bn in all_layer_names[1:]])
                axes[i, di].invert_yaxis()
                fig.colorbar(im, ax=axes[i, di])
        plt.tight_layout()
        finish_plot(show, save, os.path.join(fig_output_dir, f'patching_{metric}_per_cluster.png'), fig=fig)


# ── RDX experiment ───────────────────────────────────────────────────────────
def run_rdx(rdx_params, train_ds, acts, labels, pred_labels, inds, args, force_run=False):
    # activation_patching(visual, train_ds, inds, args, force_run=force_run)

    # Remove normalize for plotting raw images in RDX viz
    train_ds.transform = torchvision.transforms.Compose(train_ds.transform.transforms[:-1])
    image_samples = [train_ds[i]['input'] for i in tqdm(inds)]
    image_samples = torch.stack(image_samples)

    rdx_params['image_samples'] = image_samples
    rdx_params['dataset_labels'] = labels

    layer_names = list(acts.keys())
    exp_pairs1 = list(zip(layer_names[:-1], layer_names[1:]))
    exp_pairs2 = list(zip(layer_names[:-1], [layer_names[-1]] * len(layer_names[:-1])))

    data_dict = {'fig_data': {}, 'output_dict': {}}
    for ln1, ln2 in exp_pairs1:
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
                # reducer = PCA(2)
                # red0 = reducer.fit_transform(rdx_params['representations'][0])
                # reducer = PCA(2)
                # red1 = reducer.fit_transform(rdx_params['representations'][1])
                tsne = TSNE(2, perplexity=16, early_exaggeration=12, verbose=1)
                red0 = tsne.fit_transform(rdx_params['representations'][0])
                tsne = TSNE(2, perplexity=16, early_exaggeration=12, verbose=1)
                red1 = tsne.fit_transform(rdx_params['representations'][1])
                rdx_params['red0'] = red0
                rdx_params['red1'] = red1

                # TODO delete, reports the preservation ratio of top16 in reduced space
                a = torch.cdist(torch.tensor(red1), torch.tensor(red1))
                b = torch.cdist(rdx_params['representations'][1], rdx_params['representations'][1])
                topka = torch.topk(a, k=16, dim=-1, largest=False).indices
                topkb = torch.topk(b, k=16, dim=-1, largest=False).indices
                preserved = []
                for i in range(topka.shape[0]):
                    # Convert to sets and compute intersection
                    overlap = len(set(topka[i].tolist()) & set(topkb[i].tolist()))
                    preserved.append(overlap / 16)
                print('Preserved neighbor fraction:', np.mean(preserved))

                fig_data = rdx.generate_visualizations(rdx_params, output_dict, rdx_params['viz_params'])
                with open(os.path.join(method_dir, 'fig_data.pkl'), 'wb') as f:
                    pkl.dump(fig_data, f)

        data_dict['output_dict'][(ln1, ln2)] = output_dict
        data_dict['fig_data'][(ln1, ln2)] = fig_data

    return data_dict


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = parse_args(description=__doc__)
    parser.add_argument('--num_samples', type=int, default=1500,
                        help='Number of samples to use for patching experiments')
    parser.add_argument('--probe_num_samples', type=int, default=1500,)
    parser.add_argument('--force_run', action='store_true',
                        help='Force re-computation of cached activations')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = 'outputs/rsna_biomedclip/activation_patching'
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load model & dataset
    visual, preprocess = load_model(args.device)
    train_ds = load_dataset(args.data_root, preprocess)

    # # 2. Discover transformer blocks
    # block_names, block_modules = find_transformer_blocks(visual)
    # print(f'Found {len(block_names)} transformer blocks:')
    # for n in block_names:
    #     print(f'  {n}')

    # 3. Activation patching

    activation_patching(visual, train_ds, args, force_run=args.force_run)


if __name__ == '__main__':
    main()
