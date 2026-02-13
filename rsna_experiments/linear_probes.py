"""
Train per-layer linear probes on BiomedCLIP CLS-token activations (RSNA Pneumonia).

Usage:
    python -m rsna_experiments.linear_probes [--data_root PATH] [--output_dir PATH]
                                              [--batch_size N] [--num_folds K] [--device DEV]
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from src.utils.hooks import ActivationHookV2
from rsna_experiments.utils import (
    parse_args, load_model, load_dataset,
    cls_token, find_transformer_blocks, cache_activations,
)


def run_probes(activations, labels, num_folds, seed=0):
    """Train per-layer logistic regression probes with stratified K-fold CV."""
    results = {}
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    for layer_name, X in activations.items():
        fold_accs = []
        for train_idx, val_idx in kf.split(X, labels):
            clf = LogisticRegression(max_iter=1000, solver='lbfgs', verbose=False)
            clf.fit(X[train_idx], labels[train_idx])
            acc = clf.score(X[val_idx], labels[val_idx])
            fold_accs.append(acc)
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        results[layer_name] = {'mean_acc': mean_acc, 'std_acc': std_acc,
                               'fold_accs': fold_accs}
        print(f'  {layer_name:30s}  acc = {mean_acc:.4f} +/- {std_acc:.4f}')

    return results


def run_probes_full(activations, labels, seed=None):
    """Train per-layer logistic regression probes on the full dataset.

    Assumes evaluation will be done on a separate test set.
    Returns results dict with train accuracy and the fitted classifier per layer.
    """
    results = {}
    for layer_name, X in activations.items():
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', verbose=False, random_state=seed)
        clf.fit(X, labels)
        train_acc = clf.score(X, labels)
        results[layer_name] = {'train_acc': train_acc, 'classifier': clf}
        print(f'  {layer_name:30s}  train_acc = {train_acc:.4f}')
    return results


def run_knn_probes_full(activations, labels, k=5, seed=None):
    """Train per-layer KNN probes on the full dataset.

    Assumes evaluation will be done on a separate test set.
    Returns results dict with train accuracy and the fitted classifier per layer.
    """
    results = {}
    for layer_name, X in activations.items():
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, labels)
        train_acc = clf.score(X, labels)
        results[layer_name] = {'train_acc': train_acc, 'classifier': clf}
        print(f'  {layer_name:30s}  train_acc = {train_acc:.4f}')
    return results


def evaluate_probes(probe_results, test_activations, test_labels, verbose=True):
    """Evaluate fitted probes on a separate test set.

    Args:
        probe_results: dict from run_probes_full or run_knn_probes_full,
                       mapping layer_name -> {'classifier': clf, ...}.
        test_activations: dict mapping layer_name -> array of test features.
        test_labels: array of test labels.

    Returns:
        dict mapping layer_name -> {'predictions': array, 'accuracy': float}.
    """
    results = {}
    for layer_name, X_test in test_activations.items():
        clf = probe_results[layer_name]['classifier']
        preds = clf.predict(X_test)
        acc = np.mean(preds == test_labels)
        results[layer_name] = {'predictions': preds, 'accuracy': acc}
        if verbose:
            print(f'  {layer_name:30s}  test_acc = {acc:.4f}')
    return results


def run_knn_probes(activations, labels, num_folds, k=5, seed=0):
    """Train per-layer KNN probes with stratified K-fold CV."""
    results = {}
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    for layer_name, X in activations.items():
        fold_accs = []
        for train_idx, val_idx in kf.split(X, labels):
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X[train_idx], labels[train_idx])
            acc = clf.score(X[val_idx], labels[val_idx])
            fold_accs.append(acc)
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        results[layer_name] = {'mean_acc': mean_acc, 'std_acc': std_acc,
                               'fold_accs': fold_accs}
        print(f'  {layer_name:30s}  acc = {mean_acc:.4f} +/- {std_acc:.4f}')

    return results


def main():
    parser = parse_args(description=__doc__)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--knn_k', type=int, default=5,
                        help='Number of neighbors for KNN probes')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = 'outputs/rsna_biomedclip/linear_probes'
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load model
    visual, preprocess = load_model(args.device)

    # 2. Discover transformer blocks and register CLS-token hooks
    block_names, block_modules = find_transformer_blocks(visual)
    print(f'Found {len(block_names)} transformer blocks:')
    for n in block_names:
        print(f'  {n}')

    # 3. Load dataset & cache activations
    train_ds = load_dataset(args.data_root, preprocess)
    cache_path = os.path.join(args.output_dir, 'train_activations.pt')

    if os.path.exists(cache_path):
        print(f'Loading cached activations from {cache_path}')
        cached = torch.load(cache_path, weights_only=False)
        train_acts = {k: v.numpy() if isinstance(v, torch.Tensor) else v
                      for k, v in cached['activations'].items()}
        train_labels = (cached['labels'].numpy()
                        if isinstance(cached['labels'], torch.Tensor)
                        else cached['labels'])
    else:
        hook = ActivationHookV2(move_to_cpu_in_hook=True)
        hook.register_hooks(block_names, block_modules, post_activation_fn=cls_token)
        print('Building train dataloader ...')

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        print('Caching train activations ...')
        train_acts, train_labels = cache_activations(
            visual, train_loader, hook, args.device
        )

        # Save to disk
        torch.save({
            'activations': {k: torch.from_numpy(v) for k, v in train_acts.items()},
            'labels': torch.from_numpy(train_labels),
        }, cache_path)
        print(f'Saved cached activations to {cache_path}')
        hook.remove_hooks()

    # 4. Per-layer linear probes (K-fold CV on train split)
    linear_results_path = os.path.join(args.output_dir, 'probe_results.pt')
    if os.path.exists(linear_results_path):
        print(f'Linear probe results already exist at {linear_results_path}, skipping.')
        results = torch.load(linear_results_path, weights_only=False)
        for layer_name, res in results.items():
            print(f'  {layer_name:30s}  acc = {res["mean_acc"]:.4f} +/- {res["std_acc"]:.4f}')
    else:
        print(f'\nTraining linear probes ({args.num_folds}-fold CV) ...')
        results = run_probes(train_acts, train_labels, args.num_folds)
        torch.save(results, linear_results_path)
        print(f'\nResults saved to {linear_results_path}')

    # 5. Per-layer KNN probes (K-fold CV on train split)
    knn_results_path = os.path.join(args.output_dir, 'knn_probe_results.pt')
    if os.path.exists(knn_results_path):
        print(f'KNN probe results already exist at {knn_results_path}, skipping.')
        knn_results = torch.load(knn_results_path, weights_only=False)
        for layer_name, res in knn_results.items():
            print(f'  {layer_name:30s}  acc = {res["mean_acc"]:.4f} +/- {res["std_acc"]:.4f}')
    else:
        print(f'\nTraining KNN probes (k={args.knn_k}, {args.num_folds}-fold CV) ...')
        knn_results = run_knn_probes(train_acts, train_labels, args.num_folds, k=args.knn_k)
        torch.save(knn_results, knn_results_path)
        print(f'\nResults saved to {knn_results_path}')

    # 6. Write results summary to .txt
    txt_path = os.path.join(args.output_dir, 'probe_results.txt')
    with open(txt_path, 'w') as f:
        f.write(f'Linear Probes ({args.num_folds}-fold CV)\n')
        f.write('=' * 60 + '\n')
        for layer_name, res in results.items():
            f.write(f'  {layer_name:30s}  acc = {res["mean_acc"]:.4f} +/- {res["std_acc"]:.4f}\n')

        f.write(f'\nKNN Probes (k={args.knn_k}, {args.num_folds}-fold CV)\n')
        f.write('=' * 60 + '\n')
        for layer_name, res in knn_results.items():
            f.write(f'  {layer_name:30s}  acc = {res["mean_acc"]:.4f} +/- {res["std_acc"]:.4f}\n')
    print(f'\nText summary saved to {txt_path}')


if __name__ == '__main__':
    main()
