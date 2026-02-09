"""
Cache BiomedCLIP activations on RSNA Pneumonia and train per-layer linear probes.

Usage:
    python rsna_biomedclip_probes.py [--data_root PATH] [--output_dir PATH]
                                     [--batch_size N] [--num_folds K] [--device DEV]
"""

import argparse
import os

import numpy as np
import torch
from open_clip import create_model_from_pretrained
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.rsna_pneumonia import RSNAPneumoniaDataset
from src.utils.hooks import ActivationHookV2
from src.rdx import RDX
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torchvision
from timm.models.vision_transformer import VisionTransformer, Block

# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', type=str,
                   default='/media/nkondapa/SSD2/data/RSNA',
                   help='Path to RSNA dataset root')
    p.add_argument('--output_dir', type=str, default='outputs/rsna_biomedclip',
                   help='Where to save cached activations and results')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_folds', type=int, default=5)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────
def cls_token(x):
    """Post-activation: extract CLS token (index 0) from ViT block output."""
    return x[:, 0, :].clone()


def find_transformer_blocks(visual_encoder):
    """Return (names, modules) for every transformer block in the ViT."""
    names, modules = [], []
    for name, module in visual_encoder.named_modules():
        # open_clip ViT: trunk.blocks.0, trunk.blocks.1, …
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

def activation_patching(visual, train_ds, inds, args, force_run=False):
    # Activation patching

    dataloader = DataLoader(torch.utils.data.Subset(train_ds, inds), batch_size=args.batch_size, shuffle=False,
                            num_workers=4, )

    # TODO
    '''
        Collect Activations
        1) run forward pass on subset with collection hooks at
            1) attn head outputs N, H, S, D
            3) Block outputs N, S, D
        
        Patch one block at a time 
        1) For each block, patch each attention head output with mean activations for all inputs
        
        Metric - at each block, measure the top 16 nearest neighbor change for each sample (N)
    '''


    # cache activations at target layers on subset for act patching experiments
    if not os.path.exists(os.path.join(args.output_dir, 'full_cache_subset.pkl')) or force_run:
        # TODO
        with open(os.path.join(args.output_dir, 'full_cache_subset.pkl'), 'wb') as f:
            pkl.dump(dict(activations=activations, labels=labels), f)
    else:
        with open(os.path.join(args.output_dir, 'full_cache_subset.pkl'), 'rb') as f:
            cached = pkl.load(f)
            activations = cached['activations']
            labels = cached['labels']




# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load model
    print('Loading BiomedCLIP …')
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    visual = model.visual.to(args.device).eval().requires_grad_(False)

    # 2. Discover transformer blocks and register hooks
    block_names, block_modules = find_transformer_blocks(visual)
    print(f'Found {len(block_names)} transformer blocks:')
    for n in block_names:
        print(f'  {n}')

    train_ds = RSNAPneumoniaDataset(
        root=args.data_root, split='train', transform=preprocess, binary=True
    )
    # 3. Load dataset & cache activations
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
        print('Building train dataloader …')

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        print('Caching train activations …')
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

    # # 4. Per-layer linear probes (K-fold CV on train split)
    # print(f'\nTraining linear probes ({args.num_folds}-fold CV) …')
    # results = run_probes(train_acts, train_labels, args.num_folds)
    #
    # # Save results
    # results_path = os.path.join(args.output_dir, 'probe_results.pt')
    # torch.save(results, results_path)
    # print(f'\nResults saved to {results_path}')

    if os.path.exists(os.path.join(args.output_dir, 'probe_results.pt')):
        results = torch.load(os.path.join(args.output_dir, 'probe_results.pt'), weights_only=False)
        print("\nPer-layer probe results:")
        for layer_name, res in results.items():
            print(f'  {layer_name:30s}  acc = {res["mean_acc"]:.4f} +/- {res["std_acc"]:.4f}')

    rdx_params = {
        "method": "rdx",
        "method_name": "rdx_nb_lb_spectral",
        "sim_function": "neighborhood",
        "diff_function": "locally_biased",
        "clustering_method": "spectral",
        "add_null_cluster": True,
        "gamma_scale": None,
        "gamma": 0.05,
        "beta": 5,
        "seed": 0,
        "guidance": None,
        "n_clusters": 5,
        "viz_params": {
            "show": False,
            "save": True,
            "null_thresh": 1.5,
            "num_samples": 9,
            "grid_size": "3x3",
            "skip_low_affinity_for_summary": False,
            "cluster_sample_strategy": "maximize_total_neighborhood_affinity"
        }
    }

    force_run = False
    num_samples = 2500
    rng = np. random.default_rng(seed=rdx_params['seed'])
    inds = rng.choice(len(train_acts['trunk.blocks.0']), num_samples, replace=False)

    activation_patching(visual, train_ds, inds, args, force_run=force_run)

    # Remove normalize for plotting raw images in RDX viz
    train_ds.transform = torchvision.transforms.Compose(train_ds.transform.transforms[:-1])
    image_samples = [train_ds[i]['input'] for i in tqdm(inds)]
    image_samples = torch.stack(image_samples)
    rdx_params['image_samples'] = image_samples
    rdx_params['dataset_labels'] = train_labels[inds]

    layer_names = list(train_acts.keys())
    exp_pairs1 = list(zip(layer_names[:-1], layer_names[1:]))
    exp_pairs2 = list(zip(layer_names[:-1], [layer_names[-1]] * len(layer_names[:-1])))

    ln1, ln2 = exp_pairs1[0]

    for ln1, ln2 in exp_pairs1 + exp_pairs2:
        print(f"Running RDX for {ln1} vs. {ln2} …")
        method_dir = os.path.join(args.output_dir, 'rdx_outputs', f"rdx_{ln1}_vs_{ln2}")
        os.makedirs(method_dir, exist_ok=True)

        rdx_params['representations'] = [torch.tensor(train_acts[ln1][inds]), torch.tensor(train_acts[ln2][inds])]
        rdx = RDX()
        # red0 = PCA(2).fit_transform(rdx_params['representations'][0])
        red0 = TSNE(2).fit_transform(rdx_params['representations'][0])
        # red1 = PCA(2).fit_transform(rdx_params['representations'][1])
        red1 = TSNE(2).fit_transform(rdx_params['representations'][1])
        rdx_params['red0'] = red0
        rdx_params['red1'] = red1
        # with open(os.path.join(args.output_dir, 'rdx_results.pkl'), 'rb') as f:
        #     output_dict = pkl.load(f)
        if os.path.exists(os.path.join(method_dir, 'outputs.pkl')) and not force_run:
            with open(os.path.join(method_dir, 'outputs.pkl'), 'rb') as f:
                    output_dict = pkl.load(f)
        else:
            output_dict = rdx.fit(rdx_params)
            output_dict['method_dir'] = method_dir
            with open(os.path.join(method_dir, 'outputs.pkl'), 'wb') as f:
                pkl.dump(output_dict, f)

        if rdx_params.get('viz_params', None) is not None:
            fig_paths = rdx.generate_visualizations(rdx_params, output_dict, rdx_params['viz_params'])
            with open(os.path.join(args.output_dir, 'fig_paths.pkl'), 'wb') as f:
                pkl.dump(fig_paths, f)


if __name__ == '__main__':
    main()
