"""
Core functions for interactive RDX cluster visualization.

Shared interface:
    acts: {name: (N, D) ndarray}
    layer_names: [str, ...]
    rdx_data: {(ln1, ln2): {'cluster_dict': {'10': {...}, '01': {...}}, ...}}

Set ui_config['comparison_mode'] to control label semantics:
    'layerwise'    — convergence/spreading, pre-layer/post-layer (for within-model layer comparisons)
    'cross_model'  — neutral R0/R1 labels using the repr names (for comparing two different models)
"""
import base64
import io
import json
import os

import numpy as np
import torch
from umap import AlignedUMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def encode_thumbnail(pil_img, quality=70):
    """Encode a PIL image as a base64 JPEG string."""
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def compute_aligned_umap(acts, layer_names, mode='global'):
    """Compute AlignedUMAP embeddings for all layers.

    Args:
        acts: dict mapping layer name to (N, D) activation arrays.
        layer_names: list of layer names in order.
        mode: 'global' aligns all layers together using a sliding window,
              'pairwise' aligns each consecutive pair independently.

    Returns:
        list of (N, 2) arrays, one per layer. In 'pairwise' mode, each layer
        appears in two pairs (as post-layer of one and pre-layer of the next),
        so we use the post-layer embedding from the pair where it is ln2, except
        for the very first layer which uses its pre-layer embedding from pair 0.
    """
    N = acts[layer_names[0]].shape[0]
    relations = [{i: i for i in range(N)}]

    if mode == 'global':
        print('Computing AlignedUMAP (global)...')
        layer_acts = [acts[ln].astype(np.float32) for ln in layer_names]
        relations_all = [{i: i for i in range(N)} for _ in range(len(layer_names) - 1)]
        aligned = AlignedUMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                              random_state=42, alignment_window_size=1)
        embeddings = aligned.fit_transform(layer_acts, relations=relations_all)
        return embeddings  # list of (N, 2) arrays

    elif mode == 'pairwise':
        print('Computing AlignedUMAP (pairwise)...')
        embeddings = [None] * len(layer_names)
        for i in range(len(layer_names) - 1):
            ln1, ln2 = layer_names[i], layer_names[i + 1]
            pair_acts = [acts[ln1].astype(np.float32), acts[ln2].astype(np.float32)]
            # aligned = AlignedUMAP(n_components=2, n_neighbors=15, min_dist=0.1,
            #                       random_state=42, alignment_window_size=1)
            # pair_embs = aligned.fit_transform(pair_acts, relations=relations)

            pca = PCA(2)
            pair_embs = [pca.fit_transform(pair_acts[0]), pca.fit_transform(pair_acts[1])]
            # tsne = TSNE(2, perplexity=16, early_exaggeration=12, verbose=1)
            # red0 = tsne.fit_transform(pair_acts[0])
            # tsne = TSNE(2, perplexity=16, early_exaggeration=12, verbose=1)
            # red1 = tsne.fit_transform(pair_acts[1])
            # pair_embs = [red0, red1]
            if i == 0:
                embeddings[i] = pair_embs[0]
            embeddings[i + 1] = pair_embs[1]
        return embeddings

    else:
        raise ValueError(f"Unknown mode '{mode}', expected 'global' or 'pairwise'")


def precompute_neighbor_data(acts, layer_names, rdx_data, K=8, **kwargs):
    """Precompute neighbor info for each layer pair, direction, and cluster member.

    Both directions produce the same symmetric structure:
        r0_nn_in_cl / r0_nn_not_cl   — K nearest spatial NN in R0 (ln1)
        rdx_was_near / rdx_was_far    — K nearest same-cluster in clustered repr,
                                        split by scattered-repr spatial NN membership
        r1_nn_in_cl / r1_nn_not_cl   — K nearest spatial NN in R1 (ln2)

    Direction 10: clustered=R1, scattered=R0.  RDX split relative to R0 NN.
    Direction 01: clustered=R0, scattered=R1.  RDX split relative to R1 NN.
    """
    print('Precomputing neighbor data...')
    neighbor_data = {}

    for i in range(len(layer_names) - 1):
        ln1, ln2 = layer_names[i], layer_names[i + 1]
        key = (ln1, ln2)
        if key not in rdx_data:
            continue

        r0 = torch.tensor(acts[ln1], dtype=torch.float32)
        r1 = torch.tensor(acts[ln2], dtype=torch.float32)
        r0_dist = torch.cdist(r0, r0)
        r1_dist = torch.cdist(r1, r1)

        pair_key = f'{ln1}||{ln2}'
        neighbor_data[pair_key] = {'10': {}, '01': {}}

        for direction in ['10', '01']:
            cl_labels = rdx_data[key]['cluster_dict'][direction]['cluster_labels']
            unique_labels = np.unique(cl_labels)

            # Determine scattered/clustered repr
            if direction == '10':
                scattered_dist, clustered_dist = r0_dist, r1_dist
            else:
                scattered_dist, clustered_dist = r1_dist, r0_dist

            for ci in unique_labels:
                cluster_mask = cl_labels == ci
                cluster_indices = np.where(cluster_mask)[0]

                for img_idx in cluster_indices:
                    img_idx_int = int(img_idx)

                    # R0 spatial NN (K nearest in R0)
                    r0_d = r0_dist[img_idx].clone()
                    r0_d[img_idx] = float('inf')
                    r0_nn = r0_d.argsort()[:K].tolist()
                    r0_nn_in_cl = [int(j) for j in r0_nn if cl_labels[j] == ci]
                    r0_nn_not_cl = [int(j) for j in r0_nn if cl_labels[j] != ci]

                    # R1 spatial NN (K nearest in R1)
                    r1_d = r1_dist[img_idx].clone()
                    r1_d[img_idx] = float('inf')
                    r1_nn = r1_d.argsort()[:K].tolist()
                    r1_nn_in_cl = [int(j) for j in r1_nn if cl_labels[j] == ci]
                    r1_nn_not_cl = [int(j) for j in r1_nn if cl_labels[j] != ci]

                    if ci == 0:
                        # Null cluster: no RDX cluster neighbors
                        rdx_was_near = []
                        rdx_was_far = []
                    else:
                        # RDX cluster: K nearest same-cluster in clustered repr
                        cl_d = clustered_dist[img_idx].clone()
                        cl_d[img_idx] = float('inf')
                        cl_d[~torch.tensor(cluster_mask)] = float('inf')
                        rdx_nn = cl_d.argsort()[:K].tolist()
                        rdx_nn = [int(j) for j in rdx_nn if cl_d[j] < float('inf')]

                        # Split RDX group by scattered repr spatial NN membership
                        scattered_nn_set = set(r0_nn if direction == '10' else r1_nn)
                        rdx_was_near = [j for j in rdx_nn if j in scattered_nn_set]
                        rdx_was_far = [j for j in rdx_nn if j not in scattered_nn_set]

                    neighbor_data[pair_key][direction][img_idx_int] = {
                        'cluster': int(ci),
                        'r0_nn_in_cl': r0_nn_in_cl,
                        'r0_nn_not_cl': r0_nn_not_cl,
                        'rdx_was_near': rdx_was_near,
                        'rdx_was_far': rdx_was_far,
                        'r1_nn_in_cl': r1_nn_in_cl,
                        'r1_nn_not_cl': r1_nn_not_cl,
                    }

    return neighbor_data


def precompute_matrix_data(rdx_data, layer_names, K_matrix=16):
    """Precompute diff and affinity matrix values for each cluster member vs its neighbors.

    For each cluster point, computes top-K_matrix neighbors in three groups
    (L0 spatial, RDX cluster, L1 spatial) and extracts diff/am values for those pairs.

    Returns:
        matrix_data[pair_key][direction][img_idx] = {
            'cluster': int,
            'r0_nn': {'indices': [...], 'diff': [...], 'am': [...]},
            'rdx_nn': {'indices': [...], 'diff': [...], 'am': [...]},
            'r1_nn': {'indices': [...], 'diff': [...], 'am': [...]},
        }
    """
    print('Precomputing matrix data...')
    matrix_data = {}

    for i in range(len(layer_names) - 1):
        ln1, ln2 = layer_names[i], layer_names[i + 1]
        key = (ln1, ln2)
        if key not in rdx_data:
            continue

        gd = rdx_data[key].get('graph_dict', {})
        r0_dm = gd.get('r0_dm')
        r1_dm = gd.get('r1_dm')
        if r0_dm is None or r1_dm is None:
            continue

        # Ensure tensors
        if not isinstance(r0_dm, torch.Tensor):
            r0_dm = torch.tensor(r0_dm, dtype=torch.float32)
        if not isinstance(r1_dm, torch.Tensor):
            r1_dm = torch.tensor(r1_dm, dtype=torch.float32)

        pair_key = f'{ln1}||{ln2}'
        matrix_data[pair_key] = {'10': {}, '01': {}}

        dm_mat0 = gd.get(f'r0_dm')
        dm_mat1 = gd.get(f'r1_dm')
        for direction in ['10', '01']:
            cl_labels = rdx_data[key]['cluster_dict'][direction]['cluster_labels']
            unique_labels = np.unique(cl_labels)

            diff_mat = gd.get(f'diff_{direction}')
            am_mat = gd.get(f'am_{direction}')

            if diff_mat is None or am_mat is None:
                continue

            if not isinstance(diff_mat, torch.Tensor):
                diff_mat = torch.tensor(diff_mat, dtype=torch.float32)
            if not isinstance(am_mat, torch.Tensor):
                am_mat = torch.tensor(am_mat, dtype=torch.float32)
            if dm_mat0 is not None and not isinstance(dm_mat0, torch.Tensor):
                dm_mat0 = torch.tensor(dm_mat0, dtype=torch.float32)
                dm_mat1 = torch.tensor(dm_mat1, dtype=torch.float32)
            dm_mat = torch.stack([dm_mat0, dm_mat1], dim=0) if dm_mat0 is not None else None

            # Determine clustered distance matrix for RDX group
            if direction == '10':
                clustered_dm = r1_dm
            else:
                clustered_dm = r0_dm

            for ci in unique_labels:
                cluster_mask = cl_labels == ci
                cluster_indices = np.where(cluster_mask)[0]

                for img_idx in cluster_indices:
                    img_idx_int = int(img_idx)

                    # R0 spatial NN: top K_matrix nearest in r0_dm
                    r0_d = r0_dm[img_idx].clone()
                    r0_d[img_idx] = float('inf')
                    r0_nn = r0_d.argsort()[:K_matrix].tolist()

                    # R1 spatial NN: top K_matrix nearest in r1_dm
                    r1_d = r1_dm[img_idx].clone()
                    r1_d[img_idx] = float('inf')
                    r1_nn = r1_d.argsort()[:K_matrix].tolist()

                    if ci == 0:
                        # Null cluster: no RDX cluster neighbors
                        rdx_nn = []
                    else:
                        # RDX cluster NN: top K_matrix nearest same-cluster in clustered repr
                        cl_d = clustered_dm[img_idx].clone()
                        cl_d[img_idx] = float('inf')
                        cl_d[~torch.tensor(cluster_mask)] = float('inf')
                        rdx_nn = cl_d.argsort()[:K_matrix].tolist()
                        rdx_nn = [int(j) for j in rdx_nn if cl_d[j] < float('inf')]

                    def extract_values(indices):
                        if not indices:
                            return {'indices': [], 'dm': [], 'diff': [], 'am': []}
                        idx_t = torch.tensor(indices, dtype=torch.long)
                        result = {
                            'indices': [int(j) for j in indices],
                            'dm0': [round(v, 4) for v in dm_mat[0, img_idx, idx_t].tolist()] if dm_mat is not None else [],
                            'dm1': [round(v, 4) for v in dm_mat[1, img_idx, idx_t].tolist()] if dm_mat is not None else [],
                            'diff': [round(v, 4) for v in diff_mat[img_idx, idx_t].tolist()],
                            'am': [round(v, 4) for v in am_mat[img_idx, idx_t].tolist()],
                        }
                        return result

                    matrix_data[pair_key][direction][img_idx_int] = {
                        'cluster': int(ci),
                        'r0_nn': extract_values(r0_nn),
                        'rdx_nn': extract_values(rdx_nn),
                        'r1_nn': extract_values(r1_nn),
                    }

    return matrix_data


def precompute_ranking_data(rdx_data, layer_names, K_matrix=16):
    """Compute neighborhood affinity sum ranking for each cluster point.

    For each cluster point, finds the K_matrix nearest neighbors by spatial
    distance in the clustered representation, then computes the sum of the
    affinity submatrix among those neighbors (matching RDX's logic).

    Returns:
        ranking_data[pair_key][direction] = [
            {'idx': int, 'nhood_am_sum': float, 'cluster': int},
            ...  # sorted descending by nhood_am_sum
        ]
    """
    print('Precomputing ranking data...')
    ranking_data = {}

    for i in range(len(layer_names) - 1):
        ln1, ln2 = layer_names[i], layer_names[i + 1]
        key = (ln1, ln2)
        if key not in rdx_data:
            continue

        gd = rdx_data[key].get('graph_dict', {})
        pair_key = f'{ln1}||{ln2}'
        ranking_data[pair_key] = {}

        r0_dm = gd.get('r0_dm')
        r1_dm = gd.get('r1_dm')
        if r0_dm is not None and not isinstance(r0_dm, torch.Tensor):
            r0_dm = torch.tensor(r0_dm, dtype=torch.float32)
        if r1_dm is not None and not isinstance(r1_dm, torch.Tensor):
            r1_dm = torch.tensor(r1_dm, dtype=torch.float32)

        for direction in ['10', '01']:
            cl_labels = rdx_data[key]['cluster_dict'][direction]['cluster_labels']
            am_mat = gd.get(f'am_{direction}')
            diff_mat = gd.get(f'diff_{direction}')
            if am_mat is None or r0_dm is None or r1_dm is None:
                continue
            if not isinstance(am_mat, torch.Tensor):
                am_mat = torch.tensor(am_mat, dtype=torch.float32)
            if diff_mat is not None and not isinstance(diff_mat, torch.Tensor):
                diff_mat = torch.tensor(diff_mat, dtype=torch.float32)

            # Use clustered repr distance to find spatial top-K
            clustered_dm = r1_dm if direction == '10' else r0_dm

            entries = []
            unique_labels = np.unique(cl_labels)
            for ci in unique_labels:
                cluster_indices = np.where(cl_labels == ci)[0]
                for img_idx in cluster_indices:
                    # Find top-K nearest by spatial distance in clustered repr
                    d_row = clustered_dm[img_idx].clone()
                    d_row[img_idx] = float('inf')
                    topk_indices = d_row.argsort()[:K_matrix]
                    # Sum of affinity submatrix among the neighborhood (matches RDX logic)
                    nhood_am_sum = float(am_mat[topk_indices][:, topk_indices].sum())
                    # Sum of diff submatrix among the same neighborhood
                    nhood_diff_sum = float(diff_mat[topk_indices][:, topk_indices].sum()) if diff_mat is not None else 0.0
                    entries.append({
                        'idx': int(img_idx),
                        'nhood_am_sum': round(nhood_am_sum, 4),
                        'nhood_diff_sum': round(nhood_diff_sum, 4),
                        'cluster': int(ci),
                    })

            entries.sort(key=lambda e: e['nhood_am_sum'], reverse=True)
            ranking_data[pair_key][direction] = entries

    return ranking_data


def precompute_classifier_labels(acts, layer_names, labels, method='knn', K_knn=8):
    """Compute per-layer classifier predictions using KNN or a linear probe.

    Args:
        acts: dict mapping layer name to (N, D) activation arrays.
        layer_names: list of layer names.
        labels: ground-truth labels (length N).
        method: 'knn' for K-nearest-neighbor vote, 'lin' for logistic regression.
        K_knn: number of neighbors (only used when method='knn').

    Returns:
        clf_data[layer_name] = {
            'predictions': list of predicted labels,
            'correct': list of bools,
            'knn_type': 'binary' or 'multiclass',
            # binary only:
            'tp_fp_tn_fn': list of 'tp'/'fp'/'tn'/'fn',
        }
    """
    print(f'Precomputing classifier labels (method={method})...')

    labels_arr = np.asarray(labels)
    unique_labels = np.unique(labels_arr)
    knn_type = 'binary' if len(unique_labels) == 2 else 'multiclass'
    positive_class = max(unique_labels) if knn_type == 'binary' else None

    clf_data = {}
    for ln in layer_names:
        X = acts[ln].astype(np.float32)

        if method == 'knn':
            from sklearn.neighbors import NearestNeighbors
            from collections import Counter
            nn = NearestNeighbors(n_neighbors=K_knn + 1, metric='euclidean')
            nn.fit(X)
            _, indices = nn.kneighbors(X)
            neighbor_indices = indices[:, 1:]
            predictions = []
            for i in range(len(labels_arr)):
                neighbor_labels = labels_arr[neighbor_indices[i]]
                counts = Counter(neighbor_labels.tolist())
                pred = counts.most_common(1)[0][0]
                predictions.append(pred)
            predictions = np.array(predictions)

        elif method == 'lin':
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_predict
            clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
            predictions = cross_val_predict(clf, X, labels_arr, cv=5)

        else:
            raise ValueError(f"Unknown method '{method}', expected 'knn' or 'lin'")

        predictions = np.asarray(predictions)
        correct = (predictions == labels_arr).tolist()

        entry = {
            'predictions': predictions.tolist(),
            'correct': correct,
            'knn_type': knn_type,
        }

        if knn_type == 'binary':
            tp_fp_tn_fn = []
            for i in range(len(labels_arr)):
                gt = labels_arr[i]
                pred = predictions[i]
                if gt == positive_class and pred == positive_class:
                    tp_fp_tn_fn.append('tp')
                elif gt != positive_class and pred == positive_class:
                    tp_fp_tn_fn.append('fp')
                elif gt != positive_class and pred != positive_class:
                    tp_fp_tn_fn.append('tn')
                else:
                    tp_fp_tn_fn.append('fn')
            entry['tp_fp_tn_fn'] = tp_fp_tn_fn

        clf_data[ln] = entry

    return clf_data


# Keep old name as alias for backwards compatibility
precompute_knn_labels = precompute_classifier_labels


# ---------------------------------------------------------------------------
# Shared JS helper that both HTML generators inject.  Returns a function
# `L(pair)` producing all mode-dependent label strings.
# ---------------------------------------------------------------------------
_LABELS_JS = """
// Mode-dependent labels.  UI.comparison_mode is 'layerwise' or 'cross_model'.
// Each direction returns a `groups` array of 3 groups, each with:
//   {header, items: [{key, color, label}]}
// The display code iterates groups uniformly.
function L(pair) {
    const cm = UI.comparison_mode || 'cross_model';
    const K = UI.K;

    // Shared colors for the 3 group types
    const C = {
        sp_in:  '#16c79a',  // spatial NN, same cluster
        sp_out: '#888',     // spatial NN, diff cluster
        rdx_near: '#0f3460', // RDX cluster, was scattered NN
        rdx_far:  '#e9b450', // RDX cluster, was far
        sp2_in:  '#5e60ce',  // 2nd spatial NN, same cluster
        sp2_out: '#7b2d8e',  // 2nd spatial NN, diff cluster
    };

    function spatialGroup(name, inKey, outKey, colors) {
        return {
            header: name + ' Spatial Neighbors (K=' + K + ')',
            items: [
                {key: inKey,  color: colors[0], label: name + ' NN &amp; same cluster'},
                {key: outKey, color: colors[1], label: name + ' NN &amp; diff cluster'},
            ]
        };
    }

    function rdxGroup(desc) {
        return {
            header: 'RDX Cluster Neighbors (K=' + K + ')',
            items: [
                {key: 'rdx_was_near', color: C.rdx_near, label: desc + ' &amp; was scattered NN'},
                {key: 'rdx_was_far',  color: C.rdx_far,  label: desc + ' &amp; was far'},
            ]
        };
    }

    if (cm === 'layerwise') {
        // Layerwise: always show L0, RDX, L1 order for both directions
        return {
            dir10: 'Convergence (10)',
            dir01: 'Spreading (01)',
            r0Title: pair.ln1 + ' (pre)',
            r1Title: pair.ln2 + ' (post)',
            pairLabel: pair.ln1 + ' \\u2192 ' + pair.ln2,
            // dir 10 (convergence): scattered=L0, clustered=L1
            n10_groups: [
                spatialGroup(pair.ln1, 'r0_nn_in_cl', 'r0_nn_not_cl', [C.sp_in, C.sp_out]),
                {header: 'RDX Cluster Neighbors (K=' + K + ', brought together)',
                 items: [
                    {key: 'rdx_was_near', color: C.rdx_near, label: 'RDX group &amp; was ' + pair.ln1 + ' NN'},
                    {key: 'rdx_was_far',  color: C.rdx_far,  label: 'RDX group &amp; was far'},
                 ]},
                spatialGroup(pair.ln2, 'r1_nn_in_cl', 'r1_nn_not_cl', [C.sp2_in, C.sp2_out]),
            ],
            // dir 01 (spreading): scattered=L1, clustered=L0
            n01_groups: [
                spatialGroup(pair.ln1, 'r0_nn_in_cl', 'r0_nn_not_cl', [C.sp_in, C.sp_out]),
                {header: 'RDX Cluster Neighbors (K=' + K + ', spread apart)',
                 items: [
                    {key: 'rdx_was_near', color: C.rdx_near, label: 'RDX group &amp; was ' + pair.ln2 + ' NN'},
                    {key: 'rdx_was_far',  color: C.rdx_far,  label: 'RDX group &amp; was far'},
                 ]},
                spatialGroup(pair.ln2, 'r1_nn_in_cl', 'r1_nn_not_cl', [C.sp2_in, C.sp2_out]),
            ],
        };
    } else {
        // Cross-model: order is scattered, RDX, clustered
        return {
            dir10: pair.ln2 + ' vs. ' + pair.ln1 + ' (10)',
            dir01: pair.ln1 + ' vs. ' + pair.ln2 + ' (01)',
            r0Title: pair.ln1,
            r1Title: pair.ln2,
            pairLabel: pair.ln1 + ' vs. ' + pair.ln2,
            // dir 10: scattered=R0 (ln1), clustered=R1 (ln2)
            // order: R0 spatial, RDX, R1 spatial
            n10_groups: [
                spatialGroup(pair.ln1, 'r0_nn_in_cl', 'r0_nn_not_cl', [C.sp_in, C.sp_out]),
                {header: 'RDX Cluster Neighbors (K=' + K + ')',
                 items: [
                    {key: 'rdx_was_near', color: C.rdx_near, label: 'RDX group &amp; was ' + pair.ln1 + ' NN'},
                    {key: 'rdx_was_far',  color: C.rdx_far,  label: 'RDX group &amp; was far'},
                 ]},
                spatialGroup(pair.ln2, 'r1_nn_in_cl', 'r1_nn_not_cl', [C.sp2_in, C.sp2_out]),
            ],
            // dir 01: scattered=R1 (ln2), clustered=R0 (ln1)
            // order: R1 spatial, RDX, R0 spatial
            n01_groups: [
                spatialGroup(pair.ln2, 'r1_nn_in_cl', 'r1_nn_not_cl', [C.sp_in, C.sp_out]),
                {header: 'RDX Cluster Neighbors (K=' + K + ')',
                 items: [
                    {key: 'rdx_was_near', color: C.rdx_near, label: 'RDX group &amp; was ' + pair.ln2 + ' NN'},
                    {key: 'rdx_was_far',  color: C.rdx_far,  label: 'RDX group &amp; was far'},
                 ]},
                spatialGroup(pair.ln1, 'r0_nn_in_cl', 'r0_nn_not_cl', [C.sp2_in, C.sp2_out]),
            ],
        };
    }
}

function updateDirectionLabels() {
    const pair = LAYER_PAIRS[parseInt(document.getElementById('layer-pair-select').value)];
    const lb = L(pair);
    document.getElementById('dir-10-label').innerHTML = lb.dir10;
    document.getElementById('dir-01-label').innerHTML = lb.dir01;
}

// Returns {startLn, endLn, startTitle, endTitle} for animation/SBS ordering.
// In cross_model mode, both directions show convergence:
//   dir 10: clusters in ln2, scattered in ln1 -> animate ln1->ln2
//   dir 01: clusters in ln1, scattered in ln2 -> animate ln2->ln1
// In layerwise mode, always animate ln1 (pre) -> ln2 (post).
function getEndpoints(pair, direction) {
    const lb = L(pair);
    const cm = UI.comparison_mode || 'cross_model';
    if (cm === 'cross_model' && direction === '01') {
        return {startLn: pair.ln2, endLn: pair.ln1,
                startTitle: pair.ln2, endTitle: pair.ln1};
    }
    return {startLn: pair.ln1, endLn: pair.ln2,
            startTitle: lb.r0Title, endTitle: lb.r1Title};
}
"""

# ---------------------------------------------------------------------------
# Shared JS for the Matrix View tab (showMatrixAnalysis).
# Uses `MATRIX_DATA`, which has independent K_matrix.
# ---------------------------------------------------------------------------
_MATRIX_VIEW_JS = """
// -- Ranking slider --
function getCurrentRanking() {
    var pairKey = getPairKey();
    var direction = getDirection();
    return (RANKING_DATA[pairKey] && RANKING_DATA[pairKey][direction]) || [];
}

function updateRankingSlider() {
    var ranking = getCurrentRanking();
    var slider = document.getElementById('rank-slider');
    var label = document.getElementById('rank-label');
    if (ranking.length === 0) {
        slider.max = 0;
        slider.value = 0;
        label.textContent = 'No ranked points';
        return;
    }
    slider.max = ranking.length - 1;
    // If a point is currently selected, find its rank
    if (lastClickedIdx !== null) {
        var rankIdx = ranking.findIndex(function(e) { return e.idx === lastClickedIdx; });
        if (rankIdx >= 0) {
            slider.value = rankIdx;
        }
    }
    updateRankLabel();
}

function updateRankLabel() {
    var ranking = getCurrentRanking();
    var slider = document.getElementById('rank-slider');
    var label = document.getElementById('rank-label');
    var ri = parseInt(slider.value);
    if (ranking.length === 0 || ri >= ranking.length) {
        label.textContent = 'No ranked points';
        return;
    }
    var entry = ranking[ri];
    label.textContent = 'Rank ' + (ri + 1) + '/' + ranking.length +
        ' | #' + entry.idx + ' | Aff: ' + entry.nhood_am_sum.toFixed(4) +
        (entry.nhood_diff_sum !== undefined ? ' | Diff: ' + entry.nhood_diff_sum.toFixed(4) : '');
}

function selectRank(ri) {
    var ranking = getCurrentRanking();
    if (ranking.length === 0 || ri < 0 || ri >= ranking.length) return;
    var slider = document.getElementById('rank-slider');
    slider.value = ri;
    updateRankLabel();
    var entry = ranking[ri];
    lastClickedIdx = entry.idx;
    document.getElementById('save-snapshot-btn').style.display = '';
    if (activeTab === 'matrix') showMatrixAnalysis(entry.idx);
    else showNeighborAnalysis(entry.idx);
    // Update scatter highlight
    if (typeof highlightIdx !== 'undefined') {
        highlightIdx = entry.idx;
        if (typeof updateHighlight === 'function') updateHighlight();
    }
}

function stepRank(delta) {
    var slider = document.getElementById('rank-slider');
    var newVal = parseInt(slider.value) + delta;
    var ranking = getCurrentRanking();
    if (newVal < 0) newVal = 0;
    if (newVal >= ranking.length) newVal = ranking.length - 1;
    selectRank(newVal);
}

(function() {
    var slider = document.getElementById('rank-slider');
    slider.addEventListener('input', function() {
        selectRank(parseInt(this.value));
    });
})();

function diffColor(v) {
    // Blue-white-red colormap for diff values (typically in [-1, 1])
    var r, g, b;
    if (v < 0) {
        var t = Math.min(1, -v);
        r = Math.round(255 * (1 - t));
        g = Math.round(255 * (1 - t));
        b = 255;
    } else {
        var t = Math.min(1, v);
        r = 255;
        g = Math.round(255 * (1 - t));
        b = Math.round(255 * (1 - t));
    }
    return 'rgb(' + r + ',' + g + ',' + b + ')';
}

function diffTextColor(v) {
    return Math.abs(v) > 0.5 ? '#fff' : '#000';
}

function dmColor(v) {
    // Sequential orange colormap for distance values (low=light, high=dark)
    var t = Math.min(1, Math.max(0, v));
    var r = 255;
    var g = Math.round(255 * (1 - t * 0.7));
    var b = Math.round(255 * (1 - t * 0.9));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
}

function dmTextColor(v) {
    return v > 0.6 ? '#fff' : '#000';
}

function amColor(v) {
    // Sequential blue colormap for affinity values (0 to max)
    var t = Math.min(1, Math.max(0, v));
    var r = Math.round(255 * (1 - t * 0.8));
    var g = Math.round(255 * (1 - t * 0.6));
    var b = 255;
    return 'rgb(' + r + ',' + g + ',' + b + ')';
}

function amTextColor(v) {
    return v > 0.6 ? '#fff' : '#000';
}

var activeMatrixGroupKey = null;

function _updateMHL(divId, emb, indices) {
    var div = document.getElementById(divId);
    if (!div || !div.data) return;
    var ti = div.data.length - 1;  // matrix-group highlight is last trace
    Plotly.restyle(divId, {
        x: [indices.map(function(i) { return emb.x[i]; })],
        y: [indices.map(function(i) { return emb.y[i]; })],
    }, [ti]);
}

function clearMatrixHighlight() {
    activeMatrixGroupKey = null;
    ['scatter-plot', 'scatter-pre', 'scatter-post', 'scatter-anim'].forEach(function(divId) {
        var div = document.getElementById(divId);
        if (div && div.data) {
            var ti = div.data.length - 1;
            Plotly.restyle(divId, {x: [[]], y: [[]]}, [ti]);
        }
    });
    // Update button styles
    document.querySelectorAll('.matrix-highlight-btn').forEach(function(b) {
        b.classList.remove('active');
    });
}

function highlightMatrixGroup(groupKey, indices) {
    // Toggle off if clicking same group
    if (activeMatrixGroupKey === groupKey) {
        clearMatrixHighlight();
        return;
    }
    activeMatrixGroupKey = groupKey;

    // Update button styles
    document.querySelectorAll('.matrix-highlight-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.groupKey === groupKey);
    });

    var pair = LAYER_PAIRS[parseInt(document.getElementById('layer-pair-select').value)];
    var dir = getDirection();

    // Transition mode
    var ep = getEndpoints(pair, dir);
    _updateMHL('scatter-pre', EMBEDDINGS[ep.startLn], indices);
    _updateMHL('scatter-post', EMBEDDINGS[ep.endLn], indices);

    var animDiv = document.getElementById('scatter-anim');
    if (animDiv && animDiv.data) {
        var slider = document.getElementById('anim-slider');
        var t = slider ? parseInt(slider.value) / 100 : 0;
        var e0 = EMBEDDINGS[ep.startLn], e1 = EMBEDDINGS[ep.endLn];
        _updateMHL('scatter-anim', {
            x: e0.x.map(function(v, i) { return v * (1 - t) + e1.x[i] * t; }),
            y: e0.y.map(function(v, i) { return v * (1 - t) + e1.y[i] * t; })
        }, indices);
    }
}

function showMatrixAnalysis(imgIdx) {
    clearMatrixHighlight();
    updateRankingSlider();
    var pairKey = getPairKey();
    var direction = getDirection();
    var pair = LAYER_PAIRS[parseInt(document.getElementById('layer-pair-select').value)];
    var lb = L(pair);
    var md = MATRIX_DATA[pairKey] && MATRIX_DATA[pairKey][direction] && MATRIX_DATA[pairKey][direction][imgIdx];

    if (!md) {
        document.getElementById('right-content').innerHTML =
            '<div id="right-placeholder">No matrix data for this point (null cluster or no data)</div>';
        return;
    }

    var K_matrix = UI.K_matrix || 16;
    var html = '<div class="clicked-img">' +
        '<img src="' + thumbSrc(imgIdx) + '" style="cursor:pointer" onclick="showSingleModal(' + imgIdx + ')">' +
        '<div class="info">Image #' + imgIdx + ' | Label: ' + LABELS[imgIdx] + ' | Cluster: ' + md.cluster + '</div>' +
        '</div>';

    var allGroupDefs = [
        {key: 'r0_nn', label: pair.ln1 + ' Spatial Neighbors (K=' + K_matrix + ')'},
        {key: 'rdx_nn', label: 'RDX Cluster Neighbors (K=' + K_matrix + ')'},
        {key: 'r1_nn', label: pair.ln2 + ' Spatial Neighbors (K=' + K_matrix + ')'},
    ];
    // Skip RDX cluster group for null cluster points
    var groupDefs = md.cluster === 0
        ? allGroupDefs.filter(function(gd) { return gd.key !== 'rdx_nn'; })
        : allGroupDefs;

    // Compute global min/max for dm and am values across all groups for normalization
    var amMin = Infinity, amMax = -Infinity;
    var dmMin = Infinity, dmMax = -Infinity;
    groupDefs.forEach(function(gd) {
        var g = md[gd.key];
        if (g && g.am) {
            g.am.forEach(function(v) {
                if (v < amMin) amMin = v;
                if (v > amMax) amMax = v;
            });
        }
        if (g && g.dm0) {
            g.dm0.forEach(function(v) {
                if (v < dmMin) dmMin = v;
                if (v > dmMax) dmMax = v;
            });
        }
        if (g && g.dm1) {
            g.dm1.forEach(function(v) {
                if (v < dmMin) dmMin = v;
                if (v > dmMax) dmMax = v;
            });
        }
    });
    var amRange = amMax - amMin || 1;
    var dmRange = dmMax - dmMin || 1;

    groupDefs.forEach(function(gd) {
        var g = md[gd.key];
        if (!g || !g.indices || g.indices.length === 0) {
            html += '<div class="matrix-group"><h3>' + gd.label + '</h3><p style="color:#888">No neighbors</p></div>';
            return;
        }

        var indicesJson = JSON.stringify(g.indices);
        html += '<div class="matrix-group"><h3>' + gd.label +
            ' <button class="matrix-highlight-btn" data-group-key="' + gd.key +
            '" onclick="highlightMatrixGroup(&quot;' + gd.key + '&quot;,' + indicesJson +
            ')">Show on plot</button></h3>';

        // Thumbnails row
        html += '<div class="matrix-row">';
        g.indices.forEach(function(j) {
            html += '<div class="matrix-cell">' +
                '<img src="' + thumbSrc(j) + '" width="48" height="48" ' +
                'style="cursor:pointer" onclick="openModal(this.parentElement,' + j + ')">' +
                '<span class="idx-label" style="font-size:9px;color:#aaa">' + j + '</span></div>';
        });
        html += '</div>';

        // Distance row 0
        if (g.dm0 && g.dm0.length > 0) {
            html += '<div class="matrix-label">' + lb.r0Title + ' Dist</div><div class="matrix-row">';
            g.dm0.forEach(function(v) {
                var norm = (v - dmMin) / dmRange;
                var bg = dmColor(norm);
                var fg = dmTextColor(norm);
                html += '<div class="matrix-val" style="background:' + bg + ';color:' + fg + '">' + v.toFixed(2) + '</div>';
            });
            html += '</div>';
        }
        
        // Distance row 1
        if (g.dm1 && g.dm1.length > 0) {
            html += '<div class="matrix-label">' + lb.r1Title + ' Dist</div><div class="matrix-row">';
            g.dm1.forEach(function(v) {
                var norm = (v - dmMin) / dmRange;
                var bg = dmColor(norm);
                var fg = dmTextColor(norm);
                html += '<div class="matrix-val" style="background:' + bg + ';color:' + fg + '">' + v.toFixed(2) + '</div>';
            });
            html += '</div>';
        }

        // Diff row
        if (g.diff && g.diff.length > 0) {
        html += '<div class="matrix-label">diff</div><div class="matrix-row">';
        g.diff.forEach(function(v) {
            var bg = diffColor(v);
            var fg = diffTextColor(v);
            html += '<div class="matrix-val" style="background:' + bg + ';color:' + fg + '">' + v.toFixed(2) + '</div>';
        });
        html += '</div>';
        }

        // Affinity row
        // html += '<div class="matrix-label">affinity</div><div class="matrix-row">';
        // g.am.forEach(function(v) {
        //    var norm = (v - amMin) / amRange;
        //    var bg = amColor(norm);
        //    var fg = amTextColor(norm);
        //    html += '<div class="matrix-val" style="background:' + bg + ';color:' + fg + '">' + v.toFixed(2) + '</div>';
        // });
        // html += '</div>';

        html += '</div>';
    });

    document.getElementById('right-content').innerHTML = html;
}
"""



def generate_html(embeddings, layer_names, rdx_data, neighbor_data, thumbs_dir, labels,
                              output_path, ui_config=None, matrix_data=None, ranking_data=None,
                              knn_data=None, lin_data=None, lazy_load=False):
    """Generate HTML with side-by-side pre/post scatter + animated transition mode."""
    print('Generating transition HTML...')

    # Defaults (superset of single-mode defaults)
    c = {
        'body_margin': 16, 'panel_gap': 16, 'panel_padding': 16, 'panel_radius': 10,
        'left_flex': 3, 'left_min_width': 600, 'right_flex': 2, 'right_min_width': 500,
        'controls_gap': 24, 'controls_margin_bottom': 14,
        'font_controls': 16, 'font_select': 15, 'font_h2': 22, 'font_h3': 17,
        'font_legend': 14, 'font_info': 15, 'font_idx_label': 10, 'font_placeholder': 16,
        'font_plot_axis_title': 16, 'font_plot_tick': 13, 'font_plot_general': 14,
        'font_plot_legend': 13,
        'thumb_display_size': 96, 'thumb_border_width': 4, 'thumb_grid_gap': 6,
        'clicked_img_size': 180, 'clicked_img_border': 4,
        'modal_img_size': 400, 'modal_border': 3, 'modal_font_size': 16,
        'marker_null_size': 6, 'marker_null_opacity': 0.3,
        'marker_cluster_size': 10, 'marker_cluster_opacity': 0.8,
        'scatter_min_height': 600, 'scatter_height_offset': 160,
        'legend_gap': 12, 'legend_swatch': 18, 'radio_size': 16,
        'anim_duration_ms': 2000,
        'show_post_spatial_nn': True,
        'K': 8,
        'comparison_mode': 'cross_model',
    }
    if ui_config is not None:
        c.update(ui_config)

    # Prepare data (same as single mode)
    layer_pairs = []
    for i in range(len(layer_names) - 1):
        ln1, ln2 = layer_names[i], layer_names[i + 1]
        if (ln1, ln2) in rdx_data:
            layer_pairs.append({'ln1': ln1, 'ln2': ln2, 'label': f'{ln1} vs. {ln2}'})

    embedding_data = {}
    for i, ln in enumerate(layer_names):
        embedding_data[ln] = {
            'x': embeddings[i][:, 0].tolist(),
            'y': embeddings[i][:, 1].tolist(),
        }

    cluster_data = {}
    for pair in layer_pairs:
        key = (pair['ln1'], pair['ln2'])
        pair_key = f"{pair['ln1']}||{pair['ln2']}"
        cluster_data[pair_key] = {}
        for direction in ['10', '01']:
            cl_labels = rdx_data[key]['cluster_dict'][direction]['cluster_labels']
            cluster_data[pair_key][direction] = cl_labels.tolist()

    labels_list = labels.tolist() if hasattr(labels, 'tolist') else list(labels)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Interactive RDX Cluster Visualization</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: {c['body_margin']}px; background: #1a1a2e; color: #e0e0e0; }}
.container {{ display: flex; height: calc(100vh - {c['body_margin'] * 2 + 68}px); }}
.panel {{ background: #16213e; border-radius: {c['panel_radius']}px; padding: {c['panel_padding']}px; }}
#left-panel {{ flex: {c['left_flex']}; min-width: {c['left_min_width']}px; overflow-y: auto; }}
#right-panel {{ flex: {c['right_flex']}; min-width: {c['right_min_width']}px; overflow-y: auto; }}
.splitter {{ width: 6px; cursor: col-resize; background: #2a2a4a; border-radius: 3px;
    margin: 0 {c['panel_gap'] // 2}px; flex-shrink: 0; transition: background 0.15s; }}
.splitter:hover, .splitter.active {{ background: #e94560; }}
.controls {{ display: flex; gap: {c['controls_gap']}px; align-items: center; margin-bottom: {c['controls_margin_bottom']}px; flex-wrap: wrap; }}
.controls label {{ font-weight: bold; font-size: {c['font_controls']}px; }}
.controls select {{ background: #0f3460; color: #e0e0e0; border: 1px solid #533483;
    padding: 6px 12px; border-radius: 5px; font-size: {c['font_select']}px; }}
.controls input[type="radio"] {{ width: {c['radio_size']}px; height: {c['radio_size']}px; margin: 0 4px; }}
.img-group {{ margin: 14px 0; }}
.img-group h3 {{ margin: 6px 0; font-size: {c['font_h3']}px; }}
.img-grid {{ display: flex; flex-wrap: wrap; gap: {c['thumb_grid_gap']}px; }}
.img-cell {{ position: relative; cursor: pointer; }}
.img-cell img {{ display: block; }}
.img-cell .border-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    box-sizing: border-box; pointer-events: none; }}
.img-cell .idx-label {{ position: absolute; bottom: 0; right: 0; font-size: {c['font_idx_label']}px;
    background: rgba(0,0,0,0.7); color: #fff; padding: 2px 4px; }}
.clicked-img {{ text-align: center; margin-bottom: 14px; }}
.clicked-img img {{ width: {c['clicked_img_size']}px; height: {c['clicked_img_size']}px; border: {c['clicked_img_border']}px solid #e94560; }}
.clicked-img .info {{ font-size: {c['font_info']}px; margin-top: 6px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: {c['legend_gap']}px; margin: 10px 0; font-size: {c['font_legend']}px; }}
.img-modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.8); z-index: 9999; justify-content: center; align-items: center; cursor: pointer; }}
.img-modal-overlay.active {{ display: flex; }}
.modal-pair {{ display: flex; gap: 24px; align-items: center; }}
.modal-img-container {{ display: flex; flex-direction: column; align-items: center; gap: 8px; }}
.modal-img-container img {{ width: {c['modal_img_size']}px; height: {c['modal_img_size']}px; image-rendering: pixelated; border: {c['modal_border']}px solid #e0e0e0; border-radius: 6px; }}
.modal-label {{ color: #fff; font-size: {c['modal_font_size']}px;
    background: rgba(0,0,0,0.7); padding: 6px 14px; border-radius: 4px; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; }}
.legend-swatch {{ width: {c['legend_swatch']}px; height: {c['legend_swatch']}px; border-radius: 3px; }}
h2 {{ margin: 0 0 8px 0; font-size: {c['font_h2']}px; color: #e94560; }}
#right-placeholder {{ color: #888; text-align: center; padding: 60px; font-style: italic; font-size: {c['font_placeholder']}px; }}
.tab-bar {{ display: none; gap: 4px; margin-bottom: 10px; }}  /* hidden with single tab; set to flex when adding more tabs */
.tab-btn {{ background: #0f3460; color: #e0e0e0; border: 1px solid #533483; padding: 8px 18px;
    border-radius: 6px 6px 0 0; cursor: pointer; font-size: {c['font_select']}px; font-weight: bold; }}
.tab-btn.active {{ background: #533483; border-bottom-color: #16213e; }}
#ranking-bar {{ display: flex; align-items: center; gap: 8px; padding: 6px 0; margin-bottom: 6px;
    border-bottom: 1px solid #2a2a4a; }}
#ranking-bar input[type="range"] {{ flex: 1; height: 5px; cursor: pointer; accent-color: #e94560; }}
.rank-step-btn {{ background: #0f3460; color: #e0e0e0; border: 1px solid #533483; width: 28px; height: 28px;
    border-radius: 4px; cursor: pointer; font-size: 12px; display: flex; align-items: center; justify-content: center;
    padding: 0; }}
.rank-step-btn:hover {{ background: #533483; }}
#rank-label {{ font-size: {c['font_idx_label'] + 2}px; color: #aaa; white-space: nowrap; }}
.matrix-group {{ margin: 14px 0; }}
.matrix-group h3 {{ margin: 6px 0; font-size: {c['font_h3']}px; }}
.matrix-row {{ display: flex; flex-wrap: wrap; gap: 4px; align-items: flex-end; margin: 4px 0; }}
.matrix-cell {{ display: flex; flex-direction: column; align-items: center; gap: 2px; }}
.matrix-cell img {{ display: block; border-radius: 3px; }}
.matrix-val {{ width: 48px; height: 18px; border-radius: 3px; display: flex; align-items: center;
    justify-content: center; font-size: 10px; font-weight: bold; }}
.matrix-label {{ font-size: 11px; color: #aaa; margin: 4px 0 2px 0; font-style: italic; }}
.matrix-highlight-btn {{ background: #0f3460; color: #e0e0e0; border: 1px solid #533483; padding: 3px 10px;
    border-radius: 4px; cursor: pointer; font-size: 11px; margin-left: 8px; vertical-align: middle; }}
.matrix-highlight-btn:hover {{ background: #533483; }}
.matrix-highlight-btn.active {{ background: #e9b450; color: #1a1a2e; border-color: #e9b450; }}

/* Transition-mode specific */
.view-toggle {{ display: inline-flex; border-radius: 5px; overflow: hidden; margin-left: 16px; }}
.view-toggle button {{ background: #0f3460; color: #e0e0e0; border: 1px solid #533483;
    padding: 6px 16px; font-size: {c['font_select']}px; cursor: pointer; }}
.view-toggle button.active {{ background: #533483; }}
.sbs-container {{ display: flex; gap: 8px; }}
.sbs-half {{ flex: 1; }}
.sbs-half h3 {{ text-align: center; margin: 0 0 4px 0; font-size: {c['font_h3']}px; color: #e9b450; }}
#sbs-legend, #anim-legend {{ display: flex; gap: {c['legend_gap']}px; justify-content: center;
    margin-bottom: 6px; flex-wrap: wrap; }}
.sbs-legend-item {{ display: flex; align-items: center; gap: 5px; cursor: pointer;
    padding: 3px 8px; border-radius: 4px; font-size: {c['font_legend']}px; user-select: none; }}
.sbs-legend-item:hover {{ background: rgba(255,255,255,0.08); }}
.sbs-legend-item.hidden {{ opacity: 0.3; }}
#anim-controls {{ display: flex; align-items: center; gap: 12px; margin-top: 10px; padding: 8px;
    background: #0f1a30; border-radius: 6px; }}
#anim-controls button {{ background: #533483; color: #e0e0e0; border: none; padding: 8px 16px;
    border-radius: 4px; font-size: {c['font_select']}px; cursor: pointer; min-width: 70px; }}
#anim-controls button:hover {{ background: #e94560; }}
#anim-slider {{ flex: 1; height: 6px; cursor: pointer; accent-color: #e94560; }}
#anim-label {{ font-size: {c['font_info']}px; min-width: 100px; text-align: right; }}
</style>
</head>
<body>

<div class="controls">
    <label>Layer Pair:
        <select id="layer-pair-select"></select>
    </label>
    <label>Direction:
        <input type="radio" name="direction" value="10" checked> <span id="dir-10-label"></span>
        <input type="radio" name="direction" value="01"> <span id="dir-01-label"></span>
    </label>
    <div class="view-toggle">
        <button id="btn-sbs" class="active" onclick="setView('sbs')">Side-by-Side</button>
        <button id="btn-anim" onclick="setView('anim')">Animate</button>
    </div>
    <label>Color by:
        <select id="color-mode-select" onchange="onColorModeChange()">
            <option value="cluster" selected>Cluster</option>
            <option value="label">Label</option>
            <option value="diff">NBHD Diff Sum</option>
            <option value="aff">NBHD Aff Sum</option>
        </select>
    </label>
</div>

<div class="container">
    <div class="panel" id="left-panel">
        <!-- Side-by-side view -->
        <div id="sbs-view">
            <div id="sbs-legend"></div>
            <div class="sbs-container">
                <div class="sbs-half">
                    <h3 id="sbs-pre-title"></h3>
                    <div id="scatter-pre"></div>
                </div>
                <div class="sbs-half">
                    <h3 id="sbs-post-title"></h3>
                    <div id="scatter-post"></div>
                </div>
            </div>
        </div>
        <!-- Animate view -->
        <div id="anim-view" style="display:none;">
            <div id="anim-legend"></div>
            <div id="scatter-anim"></div>
            <div id="anim-controls">
                <button id="play-btn" onclick="togglePlay()">&#9654; Play</button>
                <input type="range" id="anim-slider" min="0" max="100" value="0">
                <span id="anim-label"></span>
            </div>
        </div>
    </div>
    <div class="splitter" id="splitter"></div>
    <div class="panel" id="right-panel">
        <div class="tab-bar" id="tab-bar">
            <button class="tab-btn active" data-tab="matrix" onclick="switchTab('matrix')">Matrix View</button>
        </div>
        <div id="ranking-bar">
            <button class="rank-step-btn" onclick="stepRank(-1)">&#9664;</button>
            <input type="range" id="rank-slider" min="0" max="0" value="0">
            <button class="rank-step-btn" onclick="stepRank(1)">&#9654;</button>
            <span id="rank-label"></span>
            <button id="save-snapshot-btn" onclick="saveSnapshot()" style="display:none; background:#0f3460; color:#e0e0e0; border:1px solid #533483; padding:4px 14px; border-radius:4px; cursor:pointer; font-size:12px; font-weight:bold; white-space:nowrap;">Save Snapshot</button>
        </div>
        <div id="right-content">
            <div id="right-placeholder">Click a colored cluster point to see analysis</div>
        </div>
    </div>
</div>

<div id="img-modal" class="img-modal-overlay" onclick="closeModal()">
    <div class="modal-pair" onclick="event.stopPropagation()">
        <div class="modal-img-container">
            <img id="modal-selected-img" src="">
            <div id="modal-selected-label" class="modal-label"></div>
        </div>
        <div class="modal-img-container">
            <img id="modal-neighbor-img" src="">
            <div id="modal-neighbor-label" class="modal-label"></div>
        </div>
    </div>
</div>

{'<!-- Data loaded lazily -->' if lazy_load else '<script src="neighbor_data.js"></script><script src="matrix_data.js"></script><script src="ranking_data.js"></script>'}
<script>
const THUMBS_DIR = "{thumbs_dir}";
function thumbSrc(idx) {{ return THUMBS_DIR + '/' + String(idx).padStart(4, '0') + '.jpg'; }}
const EMBEDDINGS = {json.dumps(embedding_data)};
const CLUSTER_DATA = {json.dumps(cluster_data)};
let NEIGHBOR_DATA = {'{}' if lazy_load else '_NEIGHBOR_DATA'};
let MATRIX_DATA = {'{}' if lazy_load else '_MATRIX_DATA'};
let RANKING_DATA = {'{}' if lazy_load else '_RANKING_DATA'};
const KNN_DATA = {json.dumps(knn_data or {})};
const LIN_DATA = {json.dumps(lin_data or {})};
const LABELS = {json.dumps(labels_list)};
const LAYER_PAIRS = {json.dumps(layer_pairs)};
const UI = {json.dumps(c)};

// const CLUSTER_COLORS = ['#888888', '#e94560', '#0f3460', '#533483', '#e9b450', '#16c79a'];
const CLUSTER_COLORS = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#1b9e77", "#bcbd22", "#17becf", "#7570B3FF", "#E7298AFF"];

let currentView = 'sbs';  // 'sbs' or 'anim'
let animPlaying = false;
let animRAF = null;
let highlightIdx = null;  // currently clicked point index
let activeTab = 'matrix';
let lastClickedIdx = null;
let colorMode = 'cluster';  // 'cluster', 'label', or 'diff'

// Generate distinct colors for dataset labels
const UNIQUE_LABELS = [...new Set(LABELS)].sort((a, b) => a - b);
const LABEL_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
];
const LABEL_COLOR_MAP = {{}};
UNIQUE_LABELS.forEach((lbl, i) => {{ LABEL_COLOR_MAP[lbl] = LABEL_PALETTE[i % LABEL_PALETTE.length]; }});

""" + _LABELS_JS + f"""

// Populate dropdown
const select = document.getElementById('layer-pair-select');
LAYER_PAIRS.forEach((p, i) => {{
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = L(p).pairLabel;
    select.appendChild(opt);
}});

// Rebuild KNN/LIN options in color-mode dropdown for the current layer pair
function updateColorModeOptions() {{
    const colorSel = document.getElementById('color-mode-select');
    const prev = colorSel.value;
    // Remove old knn:/lin: options
    Array.from(colorSel.options).forEach(opt => {{
        if (opt.value.startsWith('knn:') || opt.value.startsWith('lin:')) opt.remove();
    }});
    // Add options for the two layers in the current pair
    const pair = getPair();
    const layers = [pair.ln1, pair.ln2];
    layers.forEach(ln => {{
        if (KNN_DATA[ln]) {{
            const opt = document.createElement('option');
            opt.value = 'knn:' + ln;
            opt.textContent = 'KNN (' + ln + ')';
            colorSel.appendChild(opt);
        }}
    }});
    layers.forEach(ln => {{
        if (LIN_DATA[ln]) {{
            const opt = document.createElement('option');
            opt.value = 'lin:' + ln;
            opt.textContent = 'Linear (' + ln + ')';
            colorSel.appendChild(opt);
        }}
    }});
    // Restore previous selection if still valid, otherwise fall back to 'cluster'
    const validValues = Array.from(colorSel.options).map(o => o.value);
    colorSel.value = validValues.includes(prev) ? prev : 'cluster';
    colorMode = colorSel.value;
}}

function getPairKey() {{
    const idx = parseInt(select.value);
    const p = LAYER_PAIRS[idx];
    return p.ln1 + '||' + p.ln2;
}}

function getDirection() {{
    return document.querySelector('input[name="direction"]:checked').value;
}}

function getPair() {{
    return LAYER_PAIRS[parseInt(select.value)];
}}

function switchTab(tab) {{
    activeTab = tab;
    document.querySelectorAll('.tab-btn').forEach(b => {{
        b.classList.toggle('active', b.dataset.tab === tab);
    }});
    if (tab === 'neighbor') clearMatrixHighlight();
    if (lastClickedIdx !== null) {{
        if (tab === 'neighbor') showNeighborAnalysis(lastClickedIdx);
        else showMatrixAnalysis(lastClickedIdx);
    }}
}}

function imgTag(idx, borderColor) {{
    const size = UI.thumb_display_size;
    const bw = UI.thumb_border_width;
    return `<div class="img-cell" onclick="openModal(this, ${{idx}})">` +
        `<img src="${{thumbSrc(idx)}}" width="${{size}}" height="${{size}}">` +
        `<div class="border-overlay" style="border: ${{bw}}px solid ${{borderColor}};"></div>` +
        `<div class="idx-label">${{idx}}</div></div>`;
}}

// -- Image modal (click-to-enlarge + arrow key navigation) --
let modalCells = [];
let modalCellIdx = -1;

function openModal(cell, imgIdx) {{
    const grid = cell.closest('.img-grid');
    const matrixGroup = cell.closest('.matrix-group');
    if (grid) {{
        modalCells = Array.from(grid.querySelectorAll('.img-cell'));
        modalCellIdx = modalCells.indexOf(cell);
    }} else if (matrixGroup) {{
        modalCells = Array.from(matrixGroup.querySelectorAll('.matrix-cell'));
        modalCellIdx = modalCells.indexOf(cell);
    }} else {{
        modalCells = [cell];
        modalCellIdx = 0;
    }}
    showModalImage(imgIdx);
}}

function showModalImage(imgIdx) {{
    const overlay = document.getElementById('img-modal');
    // Show selected (clicked) image on the left
    const selImg = document.getElementById('modal-selected-img');
    const selLabel = document.getElementById('modal-selected-label');
    if (lastClickedIdx !== null) {{
        selImg.src = thumbSrc(lastClickedIdx);
        selLabel.textContent = 'Selected #' + lastClickedIdx + ' | Label: ' + LABELS[lastClickedIdx];
        selImg.style.display = '';
        selLabel.style.display = '';
    }} else {{
        selImg.style.display = 'none';
        selLabel.style.display = 'none';
    }}
    // Show neighbor image on the right
    document.getElementById('modal-neighbor-img').src = thumbSrc(imgIdx);
    document.getElementById('modal-neighbor-label').textContent = 'Neighbor #' + imgIdx + ' | Label: ' + LABELS[imgIdx];
    overlay.classList.add('active');
}}

function showSingleModal(imgIdx) {{
    const overlay = document.getElementById('img-modal');
    // Hide the selected (left) side
    document.getElementById('modal-selected-img').style.display = 'none';
    document.getElementById('modal-selected-label').style.display = 'none';
    // Show only the neighbor (right) side with neutral label
    document.getElementById('modal-neighbor-img').src = thumbSrc(imgIdx);
    document.getElementById('modal-neighbor-label').textContent = 'Image #' + imgIdx + ' | Label: ' + LABELS[imgIdx];
    overlay.classList.add('active');
}}

function closeModal() {{
    document.getElementById('img-modal').classList.remove('active');
    modalCells = [];
    modalCellIdx = -1;
}}

function navigateModal(delta) {{
    if (modalCells.length === 0) return;
    modalCellIdx = (modalCellIdx + delta + modalCells.length) % modalCells.length;
    const cell = modalCells[modalCellIdx];
    const imgIdx = parseInt(cell.querySelector('.idx-label').textContent);
    showModalImage(imgIdx);
}}

document.addEventListener('keydown', function(e) {{
    const overlay = document.getElementById('img-modal');
    if (!overlay.classList.contains('active')) return;
    if (e.key === 'Escape') {{ closeModal(); e.preventDefault(); }}
    else if (e.key === 'ArrowLeft') {{ navigateModal(-1); e.preventDefault(); }}
    else if (e.key === 'ArrowRight') {{ navigateModal(1); e.preventDefault(); }}
}});

// ── Save Snapshot ──────────────────────────────────────────────────

function gatherSnapshotData() {{
    var pairKey = getPairKey();
    var direction = getDirection();
    var pair = LAYER_PAIRS[parseInt(document.getElementById('layer-pair-select').value)];
    var lb = L(pair);
    var md = MATRIX_DATA[pairKey] && MATRIX_DATA[pairKey][direction] && MATRIX_DATA[pairKey][direction][lastClickedIdx];
    if (!md) return null;
    var K_matrix = UI.K_matrix || 16;

    var allGroupDefs = [
        {{key: 'r0_nn', label: lb.r0Title + ' Spatial Neighbors (K=' + K_matrix + ')'}},
        {{key: 'rdx_nn', label: 'RDX Cluster Neighbors (K=' + K_matrix + ')'}},
        {{key: 'r1_nn', label: lb.r1Title + ' Spatial Neighbors (K=' + K_matrix + ')'}},
    ];
    var groupDefs = md.cluster === 0
        ? allGroupDefs.filter(function(gd) {{ return gd.key !== 'rdx_nn'; }})
        : allGroupDefs;

    var dmMin = Infinity, dmMax = -Infinity;
    groupDefs.forEach(function(gd) {{
        var g = md[gd.key];
        if (g && g.dm0) g.dm0.forEach(function(v) {{ if (v < dmMin) dmMin = v; if (v > dmMax) dmMax = v; }});
        if (g && g.dm1) g.dm1.forEach(function(v) {{ if (v < dmMin) dmMin = v; if (v > dmMax) dmMax = v; }});
    }});
    var dmRange = dmMax - dmMin || 1;

    return {{ imgIdx: lastClickedIdx, md: md, groupDefs: groupDefs, pair: pair, lb: lb,
              pairKey: pairKey, direction: direction, K_matrix: K_matrix,
              dmMin: dmMin, dmMax: dmMax, dmRange: dmRange }};
}}

function renderSnapshotHTML(data) {{
    var md = data.md;

    var h = '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">';
    h += '<meta name="viewport" content="width=device-width, initial-scale=1.0">';
    h += '<title>RDX Snapshot — Image #' + data.imgIdx + '</title>';
    h += '<link rel="preconnect" href="https://fonts.googleapis.com">';
    h += '<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:wght@300;400;600&display=swap" rel="stylesheet">';
    h += '<style>';
    // Blog-style variables and base
    h += ':root {{ --bg: #fafaf8; --text: #1a1a1a; --muted: #6b6b6b; --accent: #2d5a8e; --border: #e2e2de; }}';
    h += '* {{ margin: 0; padding: 0; box-sizing: border-box; }}';
    h += 'body {{ background: var(--bg); color: var(--text); font-family: "Lora", Georgia, serif; font-size: 18px; line-height: 1.8; padding: 0 1.5rem; }}';
    h += 'article {{ margin: 0 auto; padding: 2rem 1rem 3rem; }}';
    h += 'h1 {{ font-family: "Lora", serif; font-size: 1.6rem; font-weight: 600; line-height: 1.25; margin-bottom: 0.5rem; letter-spacing: -0.01em; }}';
    h += '.post-meta {{ font-family: "Source Sans 3", sans-serif; font-size: 0.85rem; color: var(--muted); margin-bottom: 1.5rem; }}';
    h += '.header {{ display: flex; gap: 24px; align-items: flex-start; margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border); }}';
    h += '.header img {{ width: 160px; height: 160px; border: 2px solid var(--accent); border-radius: 4px; cursor: pointer; }}';
    h += '.header-info {{ font-family: "Source Sans 3", sans-serif; font-size: 0.95rem; color: var(--muted); line-height: 1.8; }}';
    h += '.header-info strong {{ color: var(--text); }}';
    h += 'h2 {{ font-family: "Lora", serif; font-size: 1.35rem; font-weight: 600; margin-top: 2rem; margin-bottom: 0.8rem; }}';
    h += 'h3 {{ font-family: "Source Sans 3", sans-serif; font-size: 1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-top: 2rem; margin-bottom: 0.5rem; }}';
    h += '.group {{ margin: 1.5rem 0; }}';
    h += '.thumbs {{ display: flex; flex-wrap: nowrap; gap: 4px; margin-top: 0.5rem; }}';
    h += '.cell {{ display: flex; flex-direction: column; align-items: center; flex: 1; min-width: 0; }}';
    h += '.cell img {{ width: 100%; aspect-ratio: 1; cursor: pointer; border-radius: 4px; border: 1px solid var(--border); object-fit: cover; }}';
    h += '.cell img:hover {{ border-color: var(--accent); }}';
    h += '.cell .idx {{ font-family: "Source Sans 3", sans-serif; font-size: 0.7rem; color: var(--muted); margin-top: 2px; }}';
    h += '.val-row {{ display: flex; flex-wrap: nowrap; gap: 4px; margin: 3px 0; }}';
    h += '.val-row-label {{ font-family: "Source Sans 3", sans-serif; font-size: 0.78rem; color: var(--muted); font-style: italic; margin: 6px 0 2px 0; }}';
    h += '.val {{ flex: 1; min-width: 0; height: 20px; border-radius: 3px; display: flex; align-items: center; justify-content: center; font-family: "Source Sans 3", sans-serif; font-size: 0.72rem; font-weight: 600; }}';
    // Modal
    h += '.modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 9999; justify-content: center; align-items: center; cursor: pointer; }}';
    h += '.modal-overlay.active {{ display: flex; }}';
    h += '.modal-pair {{ display: flex; gap: 24px; align-items: center; }}';
    h += '.modal-cont {{ display: flex; flex-direction: column; align-items: center; gap: 8px; }}';
    h += '.modal-cont img {{ width: 400px; height: 400px; image-rendering: pixelated; border: 3px solid #e0e0e0; border-radius: 6px; }}';
    h += '.modal-label {{ color: #fff; font-size: 16px; background: rgba(0,0,0,0.7); padding: 6px 14px; border-radius: 4px; font-family: "Source Sans 3", sans-serif; }}';
    h += 'footer {{ margin: 0 auto; padding: 1.5rem 1rem 2rem; border-top: 1px solid var(--border); font-family: "Source Sans 3", sans-serif; font-size: 0.82rem; color: var(--muted); }}';
    h += '</style></head><body>';

    // Modal
    h += '<div id="snap-modal" class="modal-overlay" onclick="closeSnapModal()">';
    h += '<div class="modal-pair" onclick="event.stopPropagation()">';
    h += '<div class="modal-cont"><img id="snap-modal-sel"><div id="snap-modal-sel-label" class="modal-label"></div></div>';
    h += '<div class="modal-cont"><img id="snap-modal-nb"><div id="snap-modal-nb-label" class="modal-label"></div></div>';
    h += '</div></div>';

    h += '<article>';
    h += '<h1>RDX Neighbor Snapshot</h1>';
    h += '<div class="post-meta">Image #' + data.imgIdx + ' &middot; Cluster ' + md.cluster + ' &middot; Direction ' + data.direction + '</div>';

    // Header with selected image
    h += '<div class="header">';
    h += '<img src="' + thumbSrc(data.imgIdx) + '" onclick="showSingle(' + data.imgIdx + ')">';
    h += '<div class="header-info">';
    h += '<strong>Image #' + data.imgIdx + '</strong><br>';
    h += 'Label: ' + LABELS[data.imgIdx] + '<br>';
    h += 'Cluster: ' + md.cluster + '<br>';
    h += 'Layer Pair: ' + data.lb.pairLabel + '<br>';
    h += 'Direction: ' + data.direction;
    h += '</div></div>';

    // Groups
    data.groupDefs.forEach(function(gd) {{
        var g = md[gd.key];
        h += '<div class="group"><h3>' + gd.label + '</h3>';
        if (!g || !g.indices || g.indices.length === 0) {{
            h += '<p style="color:var(--muted);font-style:italic">No neighbors</p></div>';
            return;
        }}

        // Thumbnails
        h += '<div class="thumbs" data-group="' + gd.key + '">';
        g.indices.forEach(function(j, ki) {{
            h += '<div class="cell" data-idx="' + j + '"><img src="' + thumbSrc(j) + '" onclick="openSnap(this,' + data.imgIdx + ',' + j + ')">';
            h += '<span class="idx">' + j + '</span></div>';
        }});
        h += '</div>';

        // Distance rows
        if (g.dm0 && g.dm0.length > 0) {{
            h += '<div class="val-row-label">' + data.lb.r0Title + ' Dist</div><div class="val-row">';
            g.dm0.forEach(function(v) {{
                var norm = (v - data.dmMin) / data.dmRange;
                h += '<div class="val" style="background:' + dmColor(norm) + ';color:' + dmTextColor(norm) + '">' + v.toFixed(2) + '</div>';
            }});
            h += '</div>';
        }}
        if (g.dm1 && g.dm1.length > 0) {{
            h += '<div class="val-row-label">' + data.lb.r1Title + ' Dist</div><div class="val-row">';
            g.dm1.forEach(function(v) {{
                var norm = (v - data.dmMin) / data.dmRange;
                h += '<div class="val" style="background:' + dmColor(norm) + ';color:' + dmTextColor(norm) + '">' + v.toFixed(2) + '</div>';
            }});
            h += '</div>';
        }}
        if (g.diff && g.diff.length > 0) {{
            h += '<div class="val-row-label">Diff</div><div class="val-row">';
            g.diff.forEach(function(v) {{
                h += '<div class="val" style="background:' + diffColor(v) + ';color:' + diffTextColor(v) + '">' + v.toFixed(2) + '</div>';
            }});
            h += '</div>';
        }}
        h += '</div>';
    }});

    h += '</article>';
    h += '<footer>Generated by RDX Interactive Cluster Visualization</footer>';

    // Inline JS for modal with arrow key navigation
    h += '<script>';
    h += 'var THUMBS_DIR = "' + THUMBS_DIR + '";';
    h += 'function tSrc(idx) {{ return THUMBS_DIR + "/" + String(idx).padStart(4, "0") + ".jpg"; }}';
    h += 'var LABELS_SNAP = ' + JSON.stringify(LABELS) + ';';
    h += 'var selIdx = ' + data.imgIdx + ';';
    h += 'var allCells = []; var curCellIdx = -1;';
    h += 'document.querySelectorAll(".thumbs .cell").forEach(function(c) {{ allCells.push(c); }});';
    h += 'function closeSnapModal() {{ document.getElementById("snap-modal").classList.remove("active"); curCellIdx = -1; }}';
    h += 'function showModalAt(ci) {{';
    h += '  if (ci < 0 || ci >= allCells.length) return;';
    h += '  curCellIdx = ci;';
    h += '  var idx = parseInt(allCells[ci].getAttribute("data-idx"));';
    h += '  var m = document.getElementById("snap-modal");';
    h += '  document.getElementById("snap-modal-sel").src = tSrc(selIdx);';
    h += '  document.getElementById("snap-modal-sel-label").textContent = "Selected #" + selIdx + " | Label: " + LABELS_SNAP[selIdx];';
    h += '  document.getElementById("snap-modal-sel").style.display = "";';
    h += '  document.getElementById("snap-modal-sel-label").style.display = "";';
    h += '  document.getElementById("snap-modal-nb").src = tSrc(idx);';
    h += '  document.getElementById("snap-modal-nb-label").textContent = "Neighbor #" + idx + " | Label: " + LABELS_SNAP[idx];';
    h += '  m.classList.add("active");';
    h += '}}';
    h += 'function openSnap(el, sIdx, nbIdx) {{';
    h += '  var cell = el.closest(".cell");';
    h += '  curCellIdx = allCells.indexOf(cell);';
    h += '  showModalAt(curCellIdx);';
    h += '}}';
    h += 'function showSingle(idx) {{';
    h += '  var m = document.getElementById("snap-modal");';
    h += '  document.getElementById("snap-modal-sel").style.display = "none";';
    h += '  document.getElementById("snap-modal-sel-label").style.display = "none";';
    h += '  document.getElementById("snap-modal-nb").src = tSrc(idx);';
    h += '  document.getElementById("snap-modal-nb-label").textContent = "Image #" + idx + " | Label: " + LABELS_SNAP[idx];';
    h += '  m.classList.add("active");';
    h += '  curCellIdx = -1;';
    h += '}}';
    h += 'document.addEventListener("keydown", function(e) {{';
    h += '  var m = document.getElementById("snap-modal");';
    h += '  if (!m.classList.contains("active")) return;';
    h += '  if (e.key === "Escape") {{ closeSnapModal(); e.preventDefault(); }}';
    h += '  else if (e.key === "ArrowLeft" && curCellIdx > 0) {{ showModalAt(curCellIdx - 1); e.preventDefault(); }}';
    h += '  else if (e.key === "ArrowRight" && curCellIdx < allCells.length - 1) {{ showModalAt(curCellIdx + 1); e.preventDefault(); }}';
    h += '}});';
    h += '<\\/script>';
    h += '</body></html>';

    return h;
}}

function saveSnapshot() {{
    var btn = document.getElementById('save-snapshot-btn');
    var origText = btn.textContent;
    btn.textContent = 'Saving...';
    btn.disabled = true;

    try {{
        var data = gatherSnapshotData();
        if (!data) {{
            alert('No matrix data available for this point.');
            return;
        }}

        // Build HTML with relative thumbs/ paths
        var htmlString = renderSnapshotHTML(data);
        var htmlBlob = new Blob([htmlString], {{type: 'text/html'}});

        // Collect all thumbnail image elements from the current page for PNG export
        var allIndices = [data.imgIdx];
        data.groupDefs.forEach(function(gd) {{
            var g = data.md[gd.key];
            if (g && g.indices) g.indices.forEach(function(j) {{ if (allIndices.indexOf(j) === -1) allIndices.push(j); }});
        }});

        // Try to build PNGs from page images via XHR blob (works on most file:// setups)
        var pngPromises = allIndices.map(function(idx) {{
            return new Promise(function(resolve) {{
                var xhr = new XMLHttpRequest();
                xhr.open('GET', thumbSrc(idx), true);
                xhr.responseType = 'blob';
                xhr.onload = function() {{
                    if (xhr.status === 0 || xhr.status === 200) resolve({{idx: idx, blob: xhr.response}});
                    else resolve({{idx: idx, blob: null}});
                }};
                xhr.onerror = function() {{ resolve({{idx: idx, blob: null}}); }};
                xhr.send();
            }});
        }});

        Promise.all(pngPromises).then(function(results) {{
            var zip = new JSZip();
            zip.file('snapshot.html', htmlBlob);

            // Add thumbnail images to zip
            var hasAny = false;
            results.forEach(function(r) {{
                if (r.blob) {{
                    zip.file('thumbs/' + String(r.idx).padStart(4, '0') + '.jpg', r.blob);
                    hasAny = true;
                }}
            }});

            if (!hasAny) {{
                // XHR failed — just download the HTML directly
                var a = document.createElement('a');
                a.href = URL.createObjectURL(htmlBlob);
                var prefix = 'snapshot_' + data.pairKey.replace('||', '_') + '_dir' + data.direction + '_img' + String(data.imgIdx).padStart(4, '0');
                a.download = prefix + '.html';
                a.click();
                URL.revokeObjectURL(a.href);
                btn.textContent = origText;
                btn.disabled = false;
                return;
            }}

            var prefix = 'snapshot_' + data.pairKey.replace('||', '_') + '_dir' + data.direction + '_img' + String(data.imgIdx).padStart(4, '0');
            zip.generateAsync({{type: 'blob'}}).then(function(zipBlob) {{
                var a = document.createElement('a');
                a.href = URL.createObjectURL(zipBlob);
                a.download = prefix + '.zip';
                a.click();
                URL.revokeObjectURL(a.href);
                btn.textContent = origText;
                btn.disabled = false;
            }});
        }});
    }} catch (err) {{
        console.error('Snapshot error:', err);
        alert('Failed to save snapshot: ' + err.message);
        btn.textContent = origText;
        btn.disabled = false;
    }}
}}

// -- Shared axis range across both repr for a pair --
function getSharedRange(pair) {{
    const e0 = EMBEDDINGS[pair.ln1];
    const e1 = EMBEDDINGS[pair.ln2];
    const allX = e0.x.concat(e1.x);
    const allY = e0.y.concat(e1.y);
    const pad = 0.05;
    const xMin = Math.min(...allX), xMax = Math.max(...allX);
    const yMin = Math.min(...allY), yMax = Math.max(...allY);
    const xPad = (xMax - xMin) * pad, yPad = (yMax - yMin) * pad;
    return {{
        xRange: [xMin - xPad, xMax + xPad],
        yRange: [yMin - yPad, yMax + yPad],
    }};
}}

// -- Build Plotly traces from embedding + cluster labels --
function buildTraces(emb, clLabels) {{
    const N = clLabels.length;
    const groups = {{}};
    for (let i = 0; i < N; i++) {{
        const cl = clLabels[i];
        if (!groups[cl]) groups[cl] = {{x: [], y: [], idx: [], color: CLUSTER_COLORS[cl] || '#888'}};
        groups[cl].x.push(emb.x[i]);
        groups[cl].y.push(emb.y[i]);
        groups[cl].idx.push(i);
    }}

    const traces = [];
    if (groups[0]) {{
        traces.push({{
            x: groups[0].x, y: groups[0].y,
            customdata: groups[0].idx,
            mode: 'markers', type: 'scatter',
            marker: {{size: UI.marker_null_size, color: '#444', opacity: UI.marker_null_opacity}},
            name: 'null',
            hovertemplate: 'idx: %{{customdata}}<br>label: %{{text}}<extra></extra>',
            text: groups[0].idx.map(i => LABELS[i].toString()),
        }});
    }}
    Object.keys(groups).filter(k => k !== '0').sort().forEach(cl => {{
        const g = groups[cl];
        traces.push({{
            x: g.x, y: g.y,
            customdata: g.idx,
            mode: 'markers', type: 'scatter',
            marker: {{size: UI.marker_cluster_size, color: g.color, opacity: UI.marker_cluster_opacity}},
            name: `cluster ${{cl}}`,
            hovertemplate: 'idx: %{{customdata}}<br>cluster: ' + cl + '<br>label: %{{text}}<extra></extra>',
            text: g.idx.map(i => LABELS[i].toString()),
        }});
    }});

    // Highlight trace (empty initially, filled when a point is clicked)
    traces.push({{
        x: [], y: [],
        mode: 'markers', type: 'scatter',
        marker: {{size: UI.marker_cluster_size + 8, color: 'rgba(0,0,0,0)',
                  line: {{color: '#fff', width: 3}}}},
        name: 'selected',
        showlegend: false,
        hoverinfo: 'skip',
    }});

    // Matrix group highlight trace (empty initially)
    traces.push({{
        x: [], y: [],
        mode: 'markers', type: 'scatter',
        marker: {{size: UI.marker_cluster_size + 6, color: 'rgba(0,0,0,0)',
                  line: {{color: '#e9b450', width: 2.5}}}},
        name: 'matrix-group',
        showlegend: false,
        hoverinfo: 'skip',
    }});

    return traces;
}}

// Build traces colored by dataset label instead of cluster
function buildLabelTraces(emb) {{
    const N = LABELS.length;
    const groups = {{}};
    for (let i = 0; i < N; i++) {{
        const lbl = LABELS[i];
        if (!groups[lbl]) groups[lbl] = {{x: [], y: [], idx: [], color: LABEL_COLOR_MAP[lbl] || '#888'}};
        groups[lbl].x.push(emb.x[i]);
        groups[lbl].y.push(emb.y[i]);
        groups[lbl].idx.push(i);
    }}

    const traces = [];
    Object.keys(groups).sort((a, b) => a - b).forEach(lbl => {{
        const g = groups[lbl];
        traces.push({{
            x: g.x, y: g.y,
            customdata: g.idx,
            mode: 'markers', type: 'scatter',
            marker: {{size: UI.marker_cluster_size, color: g.color, opacity: UI.marker_cluster_opacity}},
            name: `label ${{lbl}}`,
            hovertemplate: 'idx: %{{customdata}}<br>label: %{{text}}<extra></extra>',
            text: g.idx.map(i => LABELS[i].toString()),
        }});
    }});

    // Highlight trace
    traces.push({{
        x: [], y: [],
        mode: 'markers', type: 'scatter',
        marker: {{size: UI.marker_cluster_size + 8, color: 'rgba(0,0,0,0)',
                  line: {{color: '#fff', width: 3}}}},
        name: 'selected',
        showlegend: false,
        hoverinfo: 'skip',
    }});
    // Matrix group highlight trace
    traces.push({{
        x: [], y: [],
        mode: 'markers', type: 'scatter',
        marker: {{size: UI.marker_cluster_size + 6, color: 'rgba(0,0,0,0)',
                  line: {{color: '#e9b450', width: 2.5}}}},
        name: 'matrix-group',
        showlegend: false,
        hoverinfo: 'skip',
    }});

    return traces;
}}

// Build traces colored by a scalar value from ranking data (continuous colormap)
// valueKey: 'nhood_diff_sum' or 'nhood_am_sum'
// colorscale: Plotly colorscale name
// cmid: center value for diverging scales (null for sequential)
// title: colorbar title
function buildScalarTraces(emb, valueKey, colorscale, cmid, title) {{
    const pairKey = getPairKey();
    const direction = getDirection();
    const ranking = (RANKING_DATA[pairKey] && RANKING_DATA[pairKey][direction]) || [];
    const N = LABELS.length;

    // Build lookup from ranking entries
    const valMap = {{}};
    ranking.forEach(entry => {{ valMap[entry.idx] = entry[valueKey]; }});

    // Separate points with and without values
    const hasVal = {{x: [], y: [], idx: [], vals: []}};
    const noVal = {{x: [], y: [], idx: []}};
    for (let i = 0; i < N; i++) {{
        if (valMap[i] !== undefined) {{
            hasVal.x.push(emb.x[i]);
            hasVal.y.push(emb.y[i]);
            hasVal.idx.push(i);
            hasVal.vals.push(valMap[i]);
        }} else {{
            noVal.x.push(emb.x[i]);
            noVal.y.push(emb.y[i]);
            noVal.idx.push(i);
        }}
    }}

    const traces = [];

    // Null/no-data points
    if (noVal.x.length > 0) {{
        traces.push({{
            x: noVal.x, y: noVal.y,
            customdata: noVal.idx,
            mode: 'markers', type: 'scatter',
            marker: {{size: UI.marker_null_size, color: '#444', opacity: UI.marker_null_opacity}},
            name: 'no data',
            hovertemplate: 'idx: %{{customdata}}<br>label: %{{text}}<extra></extra>',
            text: noVal.idx.map(i => LABELS[i].toString()),
        }});
    }}

    // Colored points
    if (hasVal.x.length > 0) {{
        const markerOpts = {{
            size: UI.marker_cluster_size,
            color: hasVal.vals,
            colorscale: colorscale,
            opacity: UI.marker_cluster_opacity,
            colorbar: {{title: title, tickfont: {{color: '#e0e0e0'}}, titlefont: {{color: '#e0e0e0'}}}},
        }};
        if (cmid !== null) markerOpts.cmid = cmid;
        traces.push({{
            x: hasVal.x, y: hasVal.y,
            customdata: hasVal.idx,
            mode: 'markers', type: 'scatter',
            marker: markerOpts,
            name: title,
            hovertemplate: 'idx: %{{customdata}}<br>' + title + ': %{{marker.color:.3f}}<br>label: %{{text}}<extra></extra>',
            text: hasVal.idx.map(i => LABELS[i].toString()),
        }});
    }}

    // Highlight trace
    traces.push({{
        x: [], y: [],
        mode: 'markers', type: 'scatter',
        marker: {{size: UI.marker_cluster_size + 8, color: 'rgba(0,0,0,0)',
                  line: {{color: '#fff', width: 3}}}},
        name: 'selected',
        showlegend: false,
        hoverinfo: 'skip',
    }});
    // Matrix group highlight trace
    traces.push({{
        x: [], y: [],
        mode: 'markers', type: 'scatter',
        marker: {{size: UI.marker_cluster_size + 6, color: 'rgba(0,0,0,0)',
                  line: {{color: '#e9b450', width: 2.5}}}},
        name: 'matrix-group',
        showlegend: false,
        hoverinfo: 'skip',
    }});

    return traces;
}}

function buildDiffTraces(emb) {{
    // Blue (neg) -> White (0) -> Red (pos), centered at 0
    return buildScalarTraces(emb, 'nhood_diff_sum', [[0,'blue'],[0.5,'white'],[1,'red']], 0, 'NBHD Diff Sum');
}}

function buildAffTraces(emb) {{
    return buildScalarTraces(emb, 'nhood_am_sum', 'YlOrRd', null, 'NBHD Aff Sum');
}}

// Build traces colored by classifier (KNN or linear) result
function buildKnnTraces(emb, layerName, dataSource) {{
    const kd = dataSource[layerName];
    if (!kd) return buildLabelTraces(emb);  // fallback

    const N = LABELS.length;
    const traces = [];

    if (kd.knn_type === 'binary') {{
        // Group by tp/fp/tn/fn
        const KNN_COLORS = {{tp: '#2ca02c', fp: '#d62728', tn: '#1f77b4', fn: '#ff7f0e'}};
        const KNN_NAMES = {{tp: 'TP', fp: 'FP', tn: 'TN', fn: 'FN'}};
        const groups = {{tp: {{x:[], y:[], idx:[]}}, fp: {{x:[], y:[], idx:[]}},
                         tn: {{x:[], y:[], idx:[]}}, fn: {{x:[], y:[], idx:[]}}}};
        for (let i = 0; i < N; i++) {{
            const cat = kd.tp_fp_tn_fn[i];
            groups[cat].x.push(emb.x[i]);
            groups[cat].y.push(emb.y[i]);
            groups[cat].idx.push(i);
        }}
        ['tp', 'fp', 'tn', 'fn'].forEach(cat => {{
            const g = groups[cat];
            if (g.x.length === 0) return;
            traces.push({{
                x: g.x, y: g.y,
                customdata: g.idx,
                mode: 'markers', type: 'scatter',
                marker: {{size: UI.marker_cluster_size, color: KNN_COLORS[cat], opacity: UI.marker_cluster_opacity}},
                name: KNN_NAMES[cat],
                hovertemplate: 'idx: %{{customdata}}<br>KNN: ' + KNN_NAMES[cat] +
                    '<br>pred: %{{text}}<extra></extra>',
                text: g.idx.map(i => kd.predictions[i].toString()),
            }});
        }});
    }} else {{
        // Multiclass: group by (label, correct/incorrect)
        const groupMap = {{}};
        for (let i = 0; i < N; i++) {{
            const lbl = LABELS[i];
            const ok = kd.correct[i];
            const key = lbl + (ok ? '_correct' : '_incorrect');
            if (!groupMap[key]) groupMap[key] = {{
                x:[], y:[], idx:[], lbl: lbl, correct: ok,
                color: LABEL_COLOR_MAP[lbl] || '#888',
            }};
            groupMap[key].x.push(emb.x[i]);
            groupMap[key].y.push(emb.y[i]);
            groupMap[key].idx.push(i);
        }}
        // Sort keys so correct comes before incorrect for each label
        Object.keys(groupMap).sort().forEach(key => {{
            const g = groupMap[key];
            const suffix = g.correct ? 'correct' : 'incorrect';
            traces.push({{
                x: g.x, y: g.y,
                customdata: g.idx,
                mode: 'markers', type: 'scatter',
                marker: {{
                    size: UI.marker_cluster_size,
                    color: g.color,
                    opacity: UI.marker_cluster_opacity,
                    symbol: g.correct ? 'circle' : 'x',
                }},
                name: `${{g.lbl}} (${{suffix}})`,
                hovertemplate: 'idx: %{{customdata}}<br>label: ' + g.lbl +
                    '<br>pred: %{{text}}<br>' + suffix + '<extra></extra>',
                text: g.idx.map(i => kd.predictions[i].toString()),
            }});
        }});
    }}

    // Highlight trace
    traces.push({{
        x: [], y: [],
        mode: 'markers', type: 'scatter',
        marker: {{size: UI.marker_cluster_size + 8, color: 'rgba(0,0,0,0)',
                  line: {{color: '#fff', width: 3}}}},
        name: 'selected',
        showlegend: false,
        hoverinfo: 'skip',
    }});
    // Matrix group highlight trace
    traces.push({{
        x: [], y: [],
        mode: 'markers', type: 'scatter',
        marker: {{size: UI.marker_cluster_size + 6, color: 'rgba(0,0,0,0)',
                  line: {{color: '#e9b450', width: 2.5}}}},
        name: 'matrix-group',
        showlegend: false,
        hoverinfo: 'skip',
    }});

    return traces;
}}

// Dispatch to the right trace builder based on colorMode
function buildColoredTraces(emb, clLabels) {{
    if (colorMode.startsWith('knn:')) return buildKnnTraces(emb, colorMode.slice(4), KNN_DATA);
    if (colorMode.startsWith('lin:')) return buildKnnTraces(emb, colorMode.slice(4), LIN_DATA);
    if (colorMode === 'label') return buildLabelTraces(emb);
    if (colorMode === 'diff') return buildDiffTraces(emb);
    if (colorMode === 'aff') return buildAffTraces(emb);
    return buildTraces(emb, clLabels);
}}

function getPlotRanges(divId) {{
    const div = document.getElementById(divId);
    if (!div || !div.layout) return null;
    const xa = div.layout.xaxis, ya = div.layout.yaxis;
    if (!xa || !ya || !xa.range || !ya.range) return null;
    return {{ xRange: [...xa.range], yRange: [...ya.range] }};
}}

function applyPlotRanges(divId, ranges) {{
    if (!ranges) return;
    Plotly.relayout(divId, {{
        'xaxis.range': ranges.xRange,
        'yaxis.range': ranges.yRange,
    }});
}}

function onColorModeChange() {{
    colorMode = document.getElementById('color-mode-select').value;
    // Save current zoom ranges
    const savedPre = getPlotRanges('scatter-pre');
    const savedPost = getPlotRanges('scatter-post');
    const savedAnim = getPlotRanges('scatter-anim');
    // Re-render current view
    if (currentView === 'sbs') updateSideBySide();
    else {{ animInitialized = false; updateAnimFromSlider(); }}
    // Restore zoom ranges
    if (currentView === 'sbs') {{
        applyPlotRanges('scatter-pre', savedPre);
        applyPlotRanges('scatter-post', savedPost);
    }} else {{
        applyPlotRanges('scatter-anim', savedAnim);
    }}
    updateHighlight();
}}

function makeLayout(range, plotHeight) {{
    return {{
        xaxis: {{title: {{text: 'UMAP 1', font: {{size: UI.font_plot_axis_title}}}},
                 gridcolor: '#2a2a4a', tickfont: {{size: UI.font_plot_tick}},
                 range: range.xRange}},
        yaxis: {{title: {{text: 'UMAP 2', font: {{size: UI.font_plot_axis_title}}}},
                 gridcolor: '#2a2a4a', tickfont: {{size: UI.font_plot_tick}},
                 range: range.yRange}},
        plot_bgcolor: '#0f1a30',
        paper_bgcolor: '#16213e',
        font: {{color: '#e0e0e0', size: UI.font_plot_general}},
        legend: {{font: {{size: UI.font_plot_legend}}}},
        margin: {{l: 50, r: 10, t: 10, b: 50}},
        hovermode: 'closest',
        height: plotHeight,
    }};
}}

function attachClickHandler(divId) {{
    const div = document.getElementById(divId);
    div.removeAllListeners && div.removeAllListeners('plotly_click');
    div.on('plotly_click', function(data) {{
        const pt = data.points[0];
        if (pt.customdata !== undefined) {{
            highlightIdx = pt.customdata;
            lastClickedIdx = pt.customdata;
            document.getElementById('save-snapshot-btn').style.display = '';
            if (activeTab === 'neighbor') showNeighborAnalysis(highlightIdx);
            else showMatrixAnalysis(highlightIdx);
            updateHighlight();
        }}
    }});
}}

// -- Update the white ring on clicked point across all visible scatters --
function updateHighlight() {{
    if (highlightIdx === null) return;
    const pair = getPair();
    const direction = getDirection();
    const ep = getEndpoints(pair, direction);

    if (currentView === 'sbs') {{
        updateHighlightOnDiv('scatter-pre', EMBEDDINGS[ep.startLn], highlightIdx);
        updateHighlightOnDiv('scatter-post', EMBEDDINGS[ep.endLn], highlightIdx);
    }} else {{
        const t = parseInt(document.getElementById('anim-slider').value) / 100;
        const e0 = EMBEDDINGS[ep.startLn];
        const e1 = EMBEDDINGS[ep.endLn];
        const interpEmb = {{
            x: e0.x.map((v, i) => v * (1 - t) + e1.x[i] * t),
            y: e0.y.map((v, i) => v * (1 - t) + e1.y[i] * t),
        }};
        updateHighlightOnDiv('scatter-anim', interpEmb, highlightIdx);
    }}
}}

function updateHighlightOnDiv(divId, emb, idx) {{
    const div = document.getElementById(divId);
    if (!div || !div.data) return;
    const selIdx = div.data.length - 2;  // selected highlight is second-to-last
    Plotly.restyle(divId, {{
        x: [[emb.x[idx]]],
        y: [[emb.y[idx]]],
    }}, [selIdx]);
}}

// -- View switching --
function setView(mode) {{
    currentView = mode;
    document.getElementById('sbs-view').style.display = mode === 'sbs' ? '' : 'none';
    document.getElementById('anim-view').style.display = mode === 'anim' ? '' : 'none';
    document.getElementById('btn-sbs').className = mode === 'sbs' ? 'active' : '';
    document.getElementById('btn-anim').className = mode === 'anim' ? 'active' : '';
    if (animPlaying) stopPlay();
    updateView();
}}

// -- Side-by-side --
let sbsTraceVisible = [];

function updateSideBySide() {{
    const pair = getPair();
    const lb = L(pair);
    const pairKey = getPairKey();
    const direction = getDirection();
    const clLabels = CLUSTER_DATA[pairKey][direction];
    const range = getSharedRange(pair);
    const plotH = Math.max(UI.scatter_min_height / 1.1,
                           (window.innerHeight - UI.scatter_height_offset - 60) );

    const ep = getEndpoints(pair, direction);
    document.getElementById('sbs-pre-title').textContent = ep.startTitle;
    document.getElementById('sbs-post-title').textContent = ep.endTitle;

    const tracesPre = buildColoredTraces(EMBEDDINGS[ep.startLn], clLabels);
    const tracesPost = buildColoredTraces(EMBEDDINGS[ep.endLn], clLabels);

    tracesPre.forEach(tr => {{ tr.showlegend = false; }});
    tracesPost.forEach(tr => {{ tr.showlegend = false; }});

    const layoutPre = makeLayout(range, plotH);
    layoutPre.showlegend = false;
    const layoutPost = makeLayout(range, plotH);
    layoutPost.showlegend = false;

    Plotly.react('scatter-pre', tracesPre, layoutPre, {{responsive: true}});
    Plotly.react('scatter-post', tracesPost, layoutPost, {{responsive: true}});

    attachClickHandler('scatter-pre');
    attachClickHandler('scatter-post');

    // Build shared HTML legend (skip highlight traces and array-color traces)
    sbsTraceVisible = tracesPre.map(() => true);
    const legendDiv = document.getElementById('sbs-legend');
    let legendHtml = '';
    for (let ti = 0; ti < tracesPre.length - 2; ti++) {{
        const tr = tracesPost[ti];
        const color = tr.marker.color;
        if (typeof color !== 'string') continue;
        legendHtml += `<div class="sbs-legend-item" data-ti="${{ti}}">` +
            `<div style="width:{c['legend_swatch']}px;height:{c['legend_swatch']}px;` +
            `border-radius:3px;background:${{color}};"></div>${{tr.name}}</div>`;
    }}
    legendDiv.innerHTML = legendHtml;

    legendDiv.querySelectorAll('.sbs-legend-item').forEach(el => {{
        el.addEventListener('click', function() {{
            const ti = parseInt(this.dataset.ti);
            const nowHidden = this.classList.toggle('hidden');
            const vis = nowHidden ? 'legendonly' : true;
            sbsTraceVisible[ti] = vis;
            Plotly.restyle('scatter-pre', {{visible: vis}}, [ti]);
            Plotly.restyle('scatter-post', {{visible: vis}}, [ti]);
        }});
    }});

    updateHighlight();
}}

// -- Animate --
let animTraceIndices = null;
let animInitialized = false;

function initAnimScatter() {{
    const pair = getPair();
    const pairKey = getPairKey();
    const direction = getDirection();
    const clLabels = CLUSTER_DATA[pairKey][direction];
    const ep = getEndpoints(pair, direction);
    const e0 = EMBEDDINGS[ep.startLn];
    const range = getSharedRange(pair);
    const plotH = Math.max(UI.scatter_min_height,
                           window.innerHeight - UI.scatter_height_offset);

    // Build traces using the current color mode
    const traces = buildColoredTraces(e0, clLabels);
    traces.forEach(tr => {{ tr.showlegend = false; }});

    // Build animTraceIndices from the traces (needed for interpolation)
    animTraceIndices = [];
    const animTraceColors = [];
    const animTraceNames = [];
    traces.forEach(tr => {{
        if (tr.customdata) {{
            animTraceIndices.push(tr.customdata);
            animTraceColors.push(tr.marker.color || '#888');
            animTraceNames.push(tr.name);
        }} else {{
            animTraceIndices.push([]);
        }}
    }});

    const animLayout = makeLayout(range, plotH);
    animLayout.showlegend = false;
    Plotly.react('scatter-anim', traces, animLayout, {{responsive: true}});
    attachClickHandler('scatter-anim');

    // Build legend for anim view (skip traces with array colors like diff mode)
    const legendDiv = document.getElementById('anim-legend');
    let legendHtml = '';
    for (let ti = 0; ti < animTraceColors.length; ti++) {{
        const c_ = animTraceColors[ti];
        if (typeof c_ !== 'string') continue;
        legendHtml += `<div class="sbs-legend-item" data-ti="${{ti}}">` +
            `<div style="width:{c['legend_swatch']}px;height:{c['legend_swatch']}px;` +
            `border-radius:3px;background:${{c_}};"></div>${{animTraceNames[ti]}}</div>`;
    }}
    legendDiv.innerHTML = legendHtml;
    legendDiv.querySelectorAll('.sbs-legend-item').forEach(el => {{
        el.addEventListener('click', function() {{
            const ti = parseInt(this.dataset.ti);
            const nowHidden = this.classList.toggle('hidden');
            const vis = nowHidden ? 'legendonly' : true;
            Plotly.restyle('scatter-anim', {{visible: vis}}, [ti]);
        }});
    }});

    animInitialized = true;
}}

function updateAnimScatter(t) {{
    if (!animInitialized) {{
        initAnimScatter();
    }}

    const pair = getPair();
    const direction = getDirection();
    const ep = getEndpoints(pair, direction);
    const e0 = EMBEDDINGS[ep.startLn];
    const e1 = EMBEDDINGS[ep.endLn];

    const nTraces = animTraceIndices.length - 2;  // exclude selected + matrix-group highlight traces
    const updateX = [];
    const updateY = [];
    const traceIdxs = [];
    for (let ti = 0; ti < nTraces; ti++) {{
        const idxs = animTraceIndices[ti];
        const newX = idxs.map(i => e0.x[i] * (1 - t) + e1.x[i] * t);
        const newY = idxs.map(i => e0.y[i] * (1 - t) + e1.y[i] * t);
        updateX.push(newX);
        updateY.push(newY);
        traceIdxs.push(ti);
    }}
    Plotly.restyle('scatter-anim', {{x: updateX, y: updateY}}, traceIdxs);

    // Label — use endpoint titles
    let label;
    if (t <= 0.01) label = ep.startTitle;
    else if (t >= 0.99) label = ep.endTitle;
    else label = `t=${{t.toFixed(2)}}`;
    document.getElementById('anim-label').textContent = label;

    updateHighlight();
}}

function updateAnimFromSlider() {{
    const t = parseInt(document.getElementById('anim-slider').value) / 100;
    updateAnimScatter(t);
}}

let playStartTime = null;
let playStartValue = 0;

function togglePlay() {{
    if (animPlaying) stopPlay();
    else startPlay();
}}

function startPlay() {{
    animPlaying = true;
    document.getElementById('play-btn').innerHTML = '&#9724; Stop';
    playStartValue = parseInt(document.getElementById('anim-slider').value) / 100;
    if (playStartValue >= 0.99) playStartValue = 0;
    playStartTime = performance.now();
    animStep();
}}

function stopPlay() {{
    animPlaying = false;
    document.getElementById('play-btn').innerHTML = '&#9654; Play';
    if (animRAF) cancelAnimationFrame(animRAF);
    animRAF = null;
}}

function animStep() {{
    if (!animPlaying) return;
    const elapsed = performance.now() - playStartTime;
    const duration = UI.anim_duration_ms;
    let t = playStartValue + (elapsed / duration) * (1 - playStartValue);
    if (t >= 1) {{
        t = 1;
        stopPlay();
    }}
    document.getElementById('anim-slider').value = Math.round(t * 100);
    updateAnimScatter(t);
    if (animPlaying) animRAF = requestAnimationFrame(animStep);
}}

// -- Main update dispatcher --
function updateView() {{
    highlightIdx = null;
    lastClickedIdx = null;
    document.getElementById('save-snapshot-btn').style.display = 'none';
    animInitialized = false;
    updateColorModeOptions();
    updateDirectionLabels();
    updateRankingSlider();
    if (currentView === 'sbs') updateSideBySide();
    else {{ document.getElementById('anim-slider').value = 0; updateAnimFromSlider(); }}
    // Auto-select top ranked image
    var initRanking = getCurrentRanking();
    if (initRanking.length > 0) {{
        selectRank(0);
    }} else {{
        document.getElementById('right-content').innerHTML =
            '<div id="right-placeholder">Click a colored cluster point to see analysis</div>';
    }}
}}

// -- Neighbor analysis --
function showNeighborAnalysis(imgIdx) {{
    clearMatrixHighlight();
    updateRankingSlider();
    const pairKey = getPairKey();
    const direction = getDirection();
    const pair = getPair();
    const lb = L(pair);
    const nbData = NEIGHBOR_DATA[pairKey]?.[direction]?.[imgIdx];

    if (!nbData) {{
        document.getElementById('right-content').innerHTML =
            '<div id="right-placeholder">No neighbor data for this point (null cluster or no data)</div>';
        return;
    }}

    let html = `<div class="clicked-img">` +
        `<img src="${{thumbSrc(imgIdx)}}" style="cursor:pointer" onclick="showSingleModal(${{imgIdx}})">` +
        `<div class="info">Image #${{imgIdx}} | Label: ${{LABELS[imgIdx]}} | Cluster: ${{nbData.cluster}}</div>` +
        `</div>`;

    const groups = direction === '10' ? lb.n10_groups : lb.n01_groups;

    // Legend
    html += `<div class="legend">`;
    groups.forEach(g => {{
        const isRdxGroup = g.items.some(item => item.key === 'rdx_was_near' || item.key === 'rdx_was_far');
        if (isRdxGroup && nbData.cluster === 0) return;
        g.items.forEach(item => {{
            html += `<div class="legend-item"><div class="legend-swatch" style="background:${{item.color}}"></div>${{item.label}}</div>`;
        }});
    }});
    html += `</div>`;

    // Groups
    groups.forEach((g, gi) => {{
        // Skip RDX cluster groups for null cluster points
        const isRdxGroup = g.items.some(item => item.key === 'rdx_was_near' || item.key === 'rdx_was_far');
        if (isRdxGroup && nbData.cluster === 0) return;

        const groupIndices = [];
        g.items.forEach(item => {{
            (nbData[item.key] || []).forEach(j => {{ groupIndices.push(j); }});
        }});
        const indicesJson = JSON.stringify(groupIndices);
        html += `<div class="img-group"><h3>${{g.header}}` +
            ` <button class="matrix-highlight-btn" data-group-key="nb_${{gi}}" ` +
            `onclick="highlightMatrixGroup('nb_${{gi}}',${{indicesJson}})">Show on plot</button></h3>` +
            `<div class="img-grid">`;
        g.items.forEach(item => {{
            (nbData[item.key] || []).forEach(j => {{ html += imgTag(j, item.color); }});
        }});
        html += `</div></div>`;
    }});

    document.getElementById('right-content').innerHTML = html;
}}

""" + _MATRIX_VIEW_JS + f"""

// -- Event listeners --
select.addEventListener('change', updateView);
document.querySelectorAll('input[name="direction"]').forEach(r => {{
    r.addEventListener('change', updateView);
}});
document.getElementById('anim-slider').addEventListener('input', function() {{
    if (animPlaying) stopPlay();
    updateAnimFromSlider();
}});

// Initial render
updateDirectionLabels();
updateView();
{"" if not lazy_load else '''
// Lazy-load external data files
(function() {{
    let loaded = 0;
    const total = 3;
    function onLoaded() {{
        loaded++;
        if (loaded === total) {{
            NEIGHBOR_DATA = _NEIGHBOR_DATA;
            MATRIX_DATA = _MATRIX_DATA;
            RANKING_DATA = _RANKING_DATA;
            updateView();
        }}
    }}
    ['neighbor_data.js', 'matrix_data.js', 'ranking_data.js'].forEach(src => {{
        const s = document.createElement('script');
        s.src = src;
        s.onload = onLoaded;
        document.head.appendChild(s);
    }});
}})();
'''}

// -- Splitter drag --
(function() {{
    const splitter = document.getElementById('splitter');
    const left = document.getElementById('left-panel');
    const right = document.getElementById('right-panel');
    const container = document.querySelector('.container');
    let dragging = false;
    splitter.addEventListener('mousedown', function(e) {{
        e.preventDefault();
        dragging = true;
        splitter.classList.add('active');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    }});
    document.addEventListener('mousemove', function(e) {{
        if (!dragging) return;
        const rect = container.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const splitterW = splitter.offsetWidth;
        const minL = {c['left_min_width']};
        const minR = {c['right_min_width']};
        const leftW = Math.max(minL, Math.min(x - splitterW / 2, rect.width - minR - splitterW));
        left.style.flex = 'none';
        right.style.flex = 'none';
        left.style.width = leftW + 'px';
        right.style.width = (rect.width - leftW - splitterW) + 'px';
    }});
    document.addEventListener('mouseup', function() {{
        if (!dragging) return;
        dragging = false;
        splitter.classList.remove('active');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        // Resize all visible plotly charts
        ['scatter-pre', 'scatter-post', 'scatter-anim'].forEach(id => {{
            const el = document.getElementById(id);
            if (el && el.data) Plotly.Plots.resize(el);
        }});
    }});
}})();
</script>
</body>
</html>"""

    output_dir = os.path.dirname(output_path)

    for fname, varname, data in [('neighbor_data.js', '_NEIGHBOR_DATA', neighbor_data),
                                   ('matrix_data.js', '_MATRIX_DATA', matrix_data or {}),
                                   ('ranking_data.js', '_RANKING_DATA', ranking_data or {})]:
        fpath = os.path.join(output_dir, fname)
        with open(fpath, 'w') as jf:
            jf.write(f'const {varname} = ')
            json.dump(data, jf)
            jf.write(';\n')
        print(f'  Wrote {fname} ({os.path.getsize(fpath) / 1024 / 1024:.1f} MB)')

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Saved transition visualization to {output_path}')
