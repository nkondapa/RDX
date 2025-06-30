
import numpy as np
import tqdm
import torch
from math import ceil
import torchvision
import copy

from src.utils.parser_helper import representation_comparison_parser
from src.utils import saving, model_loader, concept_extraction_helper as ceh
from src.utils.hooks import ActivationHookV2
from src.utils.funcs import set_seed
import json
import os
from tqdm import tqdm
import pickle as pkl
import time
from sklearn.model_selection import KFold
from sklearn.decomposition import NMF
from src.utils import funcs as suf
from sklearn.decomposition import PCA
from cblearn.embedding import GNMDS
import cblearn
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from src.utils.model_loader import split_model
from matplotlib import rc, colors
rc('text.latex', preamble=r'\usepackage{color}')
import pytorch_lightning as pl
from src import cka
from src.rdx import RDX
from src import nmf_visualization_helper as mfviz
from src import kmeans_visualization_helper as kmviz
from pymf.pymf.snmf import SNMF
from pymf.pymf.cnmf import CNMF
from src.saev import SparseAutoencoder


def _batch_inference(model, dataset, batch_size=128, resize=None, device='cuda', no_grad=True):
    '''
    Code from CRAFT repository
    '''
    nb_batchs = ceil(len(dataset) / batch_size)
    start_ids = [i * batch_size for i in range(nb_batchs)]

    results = []

    context = torch.no_grad if no_grad else torch.enable_grad

    with context():
        for i in tqdm(start_ids):
            x = torch.tensor(dataset[i:i + batch_size])
            x = x.to(device)

            if resize:
                x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)

            results.append(model(x).cpu())

    results = torch.cat(results)
    return results


def shared_concept_proposals_inference(params):
    image_list = params['image_list']
    patchify = params['patchify']
    dataset_name = params['dataset_name']
    dataset_root = params['dataset_root']
    device = params['device']
    fe_outs = params['fe_outs']
    transforms = params['transforms']
    act_hooks = params['act_hooks']
    models = params['models']
    patch_size = params.get('patch_size', None)
    labels = params['labels']
    raw_data = params.get('raw_data', None)
    images_preprocessed_list = []
    if not patchify:

        for mi in range(len(fe_outs)):
            transform = transforms[mi]
            pl.seed_everything(0)
            if transform is not None:
                out = suf.load_images(image_path_list=image_list,
                                                          data_root=dataset_root,
                                                          transform=transform,
                                                          raw_data=raw_data)
                images_preprocessed = out['images_preprocessed']
                images_preprocessed_list.append(images_preprocessed.cpu())
                st = time.time()
                out = _batch_inference(models[mi], images_preprocessed, batch_size=256, device=device, no_grad=True)

                # out = models[mi].requires_grad_(True)(images_preprocessed[:2].cuda())
                # print(models[mi].layer4[1].conv2.weight.grad)
                # torch.autograd.backward(out[0], torch.randn_like(out[0]))
                # print(models[mi].layer4[1].conv2.weight.grad)

                act_hooks[mi].concatenate_layer_activations()
                print(f'Inference took {time.time() - st} seconds, {images_preprocessed.shape[0]} images')
                print(len(image_list))

    else:

        dataset_name = dataset_name
        if dataset_name == 'imagenet':
            assert transforms[0].transforms[1].size == transforms[1].transforms[1].size
            assert transforms[0].transforms[1].size[0] == 224
            # These params must be true for this function to work correctly
            num_patches = np.ceil(transforms[0].transforms[1].size[0] / patch_size) * np.ceil(
                transforms[0].transforms[1].size[0] / patch_size)
        elif dataset_name == 'nabirds':
            assert transforms[0].transforms[0].size == transforms[1].transforms[0].size
            assert transforms[0].transforms[0].size[0] == 224
            # These params must be true for this function to work correctly
            num_patches = np.ceil(transforms[0].transforms[0].size[0] / patch_size) * np.ceil(
                transforms[0].transforms[0].size[0] / patch_size)
        elif dataset_name == 'nabirds_modified':
            assert transforms[0].transforms[0].size == transforms[1].transforms[0].size
            assert transforms[0].transforms[0].size[0] == 224
            # These params must be true for this function to work correctly
            num_patches = np.ceil(transforms[0].transforms[0].size[0] / patch_size) * np.ceil(
                transforms[0].transforms[0].size[0] / patch_size)

        for mi in range(len(fe_outs)):
            transform = transforms[mi]
            out = suf.load_images(image_path_list=image_list,
                                                      data_root=dataset_root,
                                                      transform=transform)
            image_size = out['image_size']
            patches = ceh.patchify_images(out['images_preprocessed'], patch_size, strides=None)
            subsample = 8
            images_preprocessed = patches[::subsample]
            images_preprocessed.append(images_preprocessed.cpu())

            out = _batch_inference(models[mi], images_preprocessed, batch_size=256, resize=image_size, device=device)
            act_hooks[mi].concatenate_layer_activations()

        num_patches_per_image = patches.shape[0] // len(image_list)
        num_patches_per_image /= subsample
        labels = np.repeat(labels, num_patches_per_image)

    if transforms[0] is not None:
        for ti in transforms[0].transforms:
            if type(ti) == torchvision.transforms.Normalize:
                mean = ti.mean
                std = ti.std
                break
    elif transforms[1] is not None:
        for ti in transforms[1].transforms:
            if type(ti) == torchvision.transforms.Normalize:
                mean = ti.mean
                std = ti.std
                break
    else:
        mean = 0
        std = 1

    def unnormalize_base(v, mean, std):
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        return v * std + mean

    unnormalize = lambda x: unnormalize_base(x, mean, std)

    return images_preprocessed_list, labels, unnormalize


def run_cka(input_dict):
    output_dir = input_dict['output_dir']
    method_dir = os.path.join(output_dir, 'cka')
    os.makedirs(method_dir, exist_ok=True)
    representations = input_dict['representations']
    cka_val = cka.CudaCKA(device='cuda', debiased=True).linear_CKA(torch.FloatTensor(representations[0]),
                                                                       torch.FloatTensor(representations[1]))
    if input_dict.get('verbose'):
        print('CKA:', cka_val)

    output_dict = {'cka_val': cka_val, 'method_dir': method_dir}
    with open(os.path.join(method_dir, 'outputs.pkl'), 'wb') as f:
        pkl.dump(output_dict, f)
    return output_dict


def run_clf(input_dict):
    output_dir = input_dict['output_dir']
    method_dir = os.path.join(output_dir, 'clf')
    os.makedirs(method_dir, exist_ok=True)
    representations = input_dict['representations']
    num_folds = input_dict['folds']
    seed = input_dict['seed']
    indices = np.arange(representations[0].shape[0])
    np.random.shuffle(indices)
    dataset_labels = input_dict['dataset_labels']
    seed = 0
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    repr_keys = ['0', '1']
    align_reps = input_dict.get('align_representations', False)
    agreement = {"01": []}
    if align_reps:
        representations.append(input_dict['repr0_mapped'])
        representations.append(input_dict['repr1_mapped'])
        repr_keys.append('0m')
        repr_keys.append('1m')
        agreement["0m1"] = []
        agreement["01m"] = []
    accs = dict(zip(repr_keys, [[] for _ in range(len(repr_keys))]))
    clfs = dict(zip(repr_keys, [[] for _ in range(len(repr_keys))]))

    preds = np.zeros((len(indices), len(repr_keys)))
    for i, (train_index, test_index) in enumerate(kf.split(indices)):
        for ii in range(len(clfs)):
            repr_key = repr_keys[ii]
            clf = LogisticRegression(random_state=seed).fit(representations[ii][train_index], dataset_labels[train_index])
            acc = clf.score(representations[ii][test_index], dataset_labels[test_index])
            clfs[repr_key].append(clf)
            preds[test_index, ii] = clf.predict(representations[ii][test_index])
            accs[repr_key].append(acc)

        agreement["01"].append(np.mean(np.argmax(clfs['0'][i].predict_proba(representations[0][test_index]), axis=1) == np.argmax(clfs['1'][i].predict_proba(representations[1][test_index]), axis=1)))
        if align_reps:
            agreement["0m1"].append(np.mean(np.argmax(clfs['0m'][i].predict_proba(representations[0][test_index]), axis=1) == np.argmax(clfs['1'][i].predict_proba(representations[1][test_index]), axis=1)))
            agreement["01m"].append(np.mean(np.argmax(clfs['0'][i].predict_proba(representations[0][test_index]), axis=1) == np.argmax(clfs['1m'][i].predict_proba(representations[1][test_index]), axis=1)))

        # print(f'CLF: ', np.mean(np.array(accs["0"])), np.mean(np.array(accs["1"])))

    s_all = ''
    if input_dict.get('verbose', True):
        for rk in accs.keys():
            s = f'Acc {rk}: {np.mean(np.array(accs[rk]))}'
            print(s)
            s_all += s + '\n'

    with open(os.path.join(method_dir, 'outputs.txt'), 'w') as f:
        f.write(s_all)

    output_dict = {'accs': accs, 'agreement': agreement, 'method_dir': method_dir, 'preds': preds, 'labels': dataset_labels}
    with open(os.path.join(method_dir, 'outputs.pkl'), 'wb') as f:
        pkl.dump(output_dict, f)

    return output_dict


def run_rdx(input_dict, load_outputs=False):
    output_dir = input_dict['output_dir']
    sim_function = input_dict['sim_function']
    method_name = input_dict.get('method_name', f'rdx_{sim_function}')
    method_dir = os.path.join(output_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    rdx = RDX()
    if load_outputs:
        with open(os.path.join(method_dir, 'outputs.pkl'), 'rb') as f:
            output_dict = pkl.load(f)
    else:
        output_dict = rdx.fit(input_dict)
        output_dict['method_dir'] = method_dir
        output_dict['inputs'] = input_dict['method_dict']
        with open(os.path.join(method_dir, 'outputs.pkl'), 'wb') as f:
            pkl.dump(output_dict, f)

    if input_dict.get('viz_params', None) is not None:
        fig_paths = rdx.generate_visualizations(input_dict, output_dict, input_dict['viz_params'])
        with open(os.path.join(method_dir, 'fig_paths.pkl'), 'wb') as f:
            pkl.dump(fig_paths, f)

    return output_dict


def run_kmeans(input_dict, load_outputs=False):
    output_dir = input_dict['output_dir']
    method_name = input_dict.get('method_name', f'kmeans')
    method_dir = os.path.join(output_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    output_dict = {}
    representations = input_dict['representations']
    n_clusters = input_dict['n_clusters']
    seed = input_dict['seed']

    if input_dict.get('align_representations', None) is not None:
        align_dir = input_dict['align_representations']
        if align_dir == '0to1':
            representations[0] = input_dict['repr0_mapped']
            input_dict['red0'] = input_dict['r0m_red']
        elif align_dir == '1to0':
            representations[1] = input_dict['repr1_mapped']
            input_dict['red1'] = input_dict['r1m_red']

    if load_outputs:
        with open(os.path.join(method_dir, 'outputs.pkl'), 'rb') as f:
            output_dict = pkl.load(f)
    else:
        names = ['repr_0', 'repr_1']
        for ri, repr_ in enumerate(representations):
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            kmeans.fit(repr_)
            output_dict[names[ri]] = {'labels': kmeans.labels_, 'cluster_centers': kmeans.cluster_centers_}

        output_dict['method_dir'] = method_dir
        output_dict['inputs'] = input_dict['method_dict']
        with open(os.path.join(method_dir, 'outputs.pkl'), 'wb') as f:
            pkl.dump(output_dict, f)

    if input_dict.get('viz_params', None) is not None:
        fig_paths = kmviz.generate_visualizations(input_dict, output_dict, input_dict['viz_params'])
        with open(os.path.join(method_dir, 'fig_paths.pkl'), 'wb') as f:
            pkl.dump(fig_paths, f)

    return output_dict


def run_nmf(input_dict, load_outputs=False):
    output_dir = input_dict['output_dir']
    nmf_type = input_dict.get('nmf_type', 'nmf')
    method_name = input_dict.get('method_name', nmf_type)
    method_dir = os.path.join(output_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    output_dict = {}
    num_comp = input_dict['n_components']
    representations = input_dict['representations']
    seed = input_dict['seed']

    if input_dict.get('align_representations', None) is not None:
        align_dir = input_dict['align_representations']
        if align_dir == '0to1':
            representations[0] = input_dict['repr0_mapped']
            input_dict['red0'] = input_dict['r0m_red']
        elif align_dir == '1to0':
            representations[1] = input_dict['repr1_mapped']
            input_dict['red1'] = input_dict['r1m_red']

    if load_outputs:
        with open(os.path.join(method_dir, 'outputs.pkl'), 'rb') as f:
            output_dict = pkl.load(f)
    else:
        names = ['repr_0', 'repr_1']
        for ri, repr_ in enumerate(representations):
            if nmf_type == 'nmf':
                nmf = NMF(n_components=num_comp, random_state=seed)
                nmf.fit(repr_)
                U = nmf.transform(repr_)
                recon_err = np.linalg.norm(repr_ - nmf.inverse_transform(U))
                V =  nmf.components_
            elif nmf_type == 'snmf':
                nmf = SNMF(repr_.numpy().T, num_bases=num_comp, niter=100, verbose=True)
                nmf.factorize()
                U = nmf.H.T
                V = nmf.W.T
                recon = U @ V
                recon_err = torch.norm(repr_ - recon).item()
                print(recon_err)
            elif nmf_type == 'cnmf':
                nmf = CNMF(repr_.numpy().T, num_bases=num_comp, niter=100, verbose=True)
                nmf.factorize()
                U = nmf.H.T
                V = nmf.W.T
                recon = U @ V
                recon_err = torch.norm(repr_ - recon).item()
            else:
                raise ValueError(f'Unknown NMF type: {nmf_type}')

            output_dict[names[ri]] = {'recon_err': recon_err, 'U': U, 'V': V}

        output_dict['method_dir'] = method_dir
        output_dict['inputs'] = input_dict['method_dict']
        with open(os.path.join(method_dir, 'outputs.pkl'), 'wb') as f:
            pkl.dump(output_dict, f)

    if input_dict.get('viz_params', None) is not None:
        fig_paths = mfviz.generate_visualizations(input_dict, output_dict, input_dict['viz_params'])
        with open(os.path.join(method_dir, 'fig_paths.pkl'), 'wb') as f:
            pkl.dump(fig_paths, f)

    return output_dict


def run_pca(input_dict, load_outputs=False):
    output_dir = input_dict['output_dir']
    method_name = input_dict.get('method_name', f'pca')
    method_dir = os.path.join(output_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    output_dict = {}
    num_comp = input_dict['n_components']
    representations = input_dict['representations']
    seed = input_dict['seed']

    if input_dict.get('align_representations', None) is not None:
        align_dir = input_dict['align_representations']
        if align_dir == '0to1':
            representations[0] = input_dict['repr0_mapped']
            input_dict['red0'] = input_dict['r0m_red']
        elif align_dir == '1to0':
            representations[1] = input_dict['repr1_mapped']
            input_dict['red1'] = input_dict['r1m_red']

    if load_outputs:
        with open(os.path.join(method_dir, 'outputs.pkl'), 'rb') as f:
            output_dict = pkl.load(f)
    else:
        names = ['repr_0', 'repr_1']
        for ri, repr_ in enumerate(representations):
            pca = PCA(n_components=num_comp, random_state=seed)
            pca.fit(repr_)
            U = pca.transform(repr_)
            recon_err = np.linalg.norm(repr_ - pca.inverse_transform(U))
            V = pca.components_

            output_dict[names[ri]] = {'recon_err': recon_err, 'U': U, 'V': V}

        output_dict['method_dir'] = method_dir
        output_dict['inputs'] = input_dict['method_dict']
        with open(os.path.join(method_dir, 'outputs.pkl'), 'wb') as f:
            pkl.dump(output_dict, f)

    if input_dict.get('viz_params', None) is not None:
        fig_paths = mfviz.generate_visualizations(input_dict, output_dict, input_dict['viz_params'])
        with open(os.path.join(method_dir, 'fig_paths.pkl'), 'wb') as f:
            pkl.dump(fig_paths, f)

    return output_dict


def run_sae(input_dict, load_outputs=False):
    output_dir = input_dict['output_dir']
    method_name = input_dict.get('method_name', f'sae')
    method_dir = os.path.join(output_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    output_dict = {}
    representations = input_dict['representations']
    seed = input_dict['seed']
    input_dict['n_components'] = input_dict['d_sae']

    if input_dict.get('align_representations', None) is not None:
        align_dir = input_dict['align_representations']
        if align_dir == '0to1':
            representations[0] = input_dict['repr0_mapped']
            input_dict['red0'] = input_dict['r0m_red']
        elif align_dir == '1to0':
            representations[1] = input_dict['repr1_mapped']
            input_dict['red1'] = input_dict['r1m_red']

    if not load_outputs:
        names = ['repr_0', 'repr_1']
        for ri, repr_ in enumerate(representations):
            input_dict['d_input'] = repr_.shape[1]
            sae = SparseAutoencoder(input_params=input_dict)
            sae.init_b_dec((repr_ - repr_.mean() / repr_.std()))
            global_step, loss_history = sae.fit(input_dict, repr_)
            with torch.no_grad():
                sae.eval()
                repr_ = torch.tensor(repr_)
                repr_ = repr_.to(input_dict['device'])
                recon, U, _ = sae(repr_)
                U = U.cpu().numpy()
                recon_err = np.linalg.norm(repr_.cpu().numpy() - recon.cpu().numpy())
                print(recon_err)
                recon = sae.dataset.unnormalize(recon)
                recon_err = np.linalg.norm(repr_.cpu().numpy() - recon.cpu().numpy())
                print(recon_err)

            output_dict[names[ri]] = {'global_step': global_step, 'loss_history': loss_history,
                                      'U': U, 'V': sae.W_dec, 'V_bias': sae.b_dec, 'recon_err': recon_err}
            print(recon_err)
            # nmf = PCA(n_components=num_comp, random_state=seed)
            # nmf.fit(repr_)
            # U = nmf.transform(repr_)
            # recon_err = np.linalg.norm(repr_ - nmf.inverse_transform(U))
            # V =  nmf.components_

            # output_dict[names[ri]] = {'recon_err': recon_err, 'U': U, 'V': V}

        output_dict['method_dir'] = method_dir
        output_dict['inputs'] = input_dict['method_dict']
        with open(os.path.join(method_dir, 'outputs.pkl'), 'wb') as f:
            pkl.dump(output_dict, f)
    else:
        with open(os.path.join(method_dir, 'outputs.pkl'), 'rb') as f:
            output_dict = pkl.load(f)

    if input_dict.get('viz_params', None) is not None:
        fig_paths = mfviz.generate_visualizations(input_dict, output_dict, input_dict['viz_params'])
        with open(os.path.join(method_dir, 'fig_paths.pkl'), 'wb') as f:
            pkl.dump(fig_paths, f)

    return output_dict

def learn_mapping(repr_from, repr_to, params):

    num_steps = params.get('num_steps', 100)
    train_fraction = params.get('train_fraction', 0.7)
    lr = params.get('lr', 0.001)
    # use cka loss to learn mapping
    cka_loss = cka.CudaCKA(device='cuda', debiased=True)
    indices = np.arange(repr_from.shape[0])
    train_indices = np.random.choice(indices, size=int(repr_from.shape[0] * train_fraction), replace=False)
    test_indices = np.setdiff1d(indices, train_indices)
    mapping_layer = torch.nn.Linear(repr_from.shape[1], repr_from.shape[1])
    optim = torch.optim.Adam(mapping_layer.parameters(), lr=lr)
    mapping_layer.train()
    train_ckas = []
    test_ckas = []
    best_mapping_layer = None
    mapping_training_history = {}
    for _ in range(num_steps):
        optim.zero_grad()
        cka_loss_value = 1 - cka_loss.linear_CKA(mapping_layer(repr_from)[train_indices], repr_to[train_indices])
        cka_loss_value.backward()
        optim.step()
        with torch.no_grad():
            test_cka = 1 - cka_loss.linear_CKA(mapping_layer(repr_from)[test_indices], repr_to[test_indices])
        print(cka_loss_value.item(), test_cka.item())
        train_ckas.append(cka_loss_value.item())
        test_ckas.append(test_cka.item())
        if best_mapping_layer is None or test_cka < min(test_ckas):
            best_mapping_layer = mapping_layer.state_dict()
    mapping_layer.load_state_dict(best_mapping_layer)
    mapping_training_history['train_ckas'] = train_ckas
    mapping_training_history['test_ckas'] = test_ckas
    mapping_training_history['mapping_layer'] = mapping_layer

    return mapping_layer(repr_from).detach().cpu(), mapping_training_history

def main():
    parser = representation_comparison_parser()
    parser.add_argument('--skip_viz_clusters', action='store_true')
    parser.add_argument('--model_0_ckpt', type=str, default=None)
    parser.add_argument('--model_0_triplets', type=str, default=None)
    parser.add_argument('--model_1_ckpt', type=str, default=None)
    parser.add_argument('--save_m1_representation', action='store_true')
    parser.add_argument('--save_m0_representation', action='store_true')
    parser.add_argument('--create_human_evaluation_folder', action='store_true')
    args = parser.parse_args()

    with open(args.comparison_config, 'r') as f:
        config = json.load(f)

    output_root_folder = f'{args.comparison_output_root}/{args.comparison_config.split("/")[-1].replace(".json", "")}'
    os.makedirs(output_root_folder, exist_ok=True)
    if args.model_0_ckpt is not None:
        config['representation_params']['repr_0']['model_ckpt'] = args.model_0_ckpt
        output_root_folder = f'{output_root_folder}/{args.model_0_ckpt.split("/")[-1].split(".")[0]}'

    if args.model_0_triplets is not None:
        config['representation_params']['repr_0']['triplet_embedding_data'] = args.model_0_triplets
        output_root_folder = f'{output_root_folder}/{args.model_0_triplets.split("simulated_triplets/")[-1].split(".pkl")[0]}'

    representation_params = config['representation_params']
    repr_0 = representation_params['repr_0']
    repr_1 = representation_params['repr_1']
    seed = config['seed']
    set_seed(seed)
    device = 'cuda'

    # Parameters for grouping images from a large dataset (e.g. ImageNet) for concept comparison
    image_selection_params = config['image_selection']
    dataset_name = image_selection_params['dataset_name']
    dataset_root = image_selection_params['dataset_root']
    dataset_split = image_selection_params['dataset_split']
    only_last_layer = config['only_last_layer']
    transform_type = image_selection_params['transform_type']
    target_classes = image_selection_params.get('target_classes', None)
    move_to_cpu_every = config.get('move_to_cpu_every', None)
    move_to_cpu_in_hook = config.get('move_to_cpu_in_hook', None)
    classifier_guided = config.get('classifier_guided', False)
    image_group_strategy = image_selection_params['image_group_strategy']
    if image_group_strategy == 'full_dataset':
        igs_params = None
        ig_dict = suf.get_image_group(image_group_strategy, image_selection_params, igs_params, return_dataset=True)
    else:
        igs_params = {
            'data_group_params':
            {
                'dataset_name': dataset_name,
                'dataset_root': dataset_root,
                'split': dataset_split,
                'seed': seed,
                'num_images': image_selection_params['num_images'],
                'topk_group_size': image_selection_params.get('topk_group_size', None),
                'topk_prob_thresh': image_selection_params.get('topk_prob_thresh', 0),
                'target_classes': target_classes,
            },
            # for selection strategies that rely on model outputs
            'param_dicts1': {'model': (repr_0['model'], repr_0.get('model_ckpt', None))},
            'param_dicts2': {'model': (repr_1['model'], repr_1.get('model_ckpt', None))},
        }

        ig_dict = suf.get_image_group(image_group_strategy, image_selection_params, igs_params,
                                      return_dataset=True)
    image_group, labels, dataset = ig_dict['image_group'], ig_dict['labels'], ig_dict['dataset']

    if hasattr(dataset, 'data'):
        raw_data = dataset.data
    else:
        raw_data = None

    # Load models, transforms for models, and layers
    patchify = image_selection_params.get('patchify', False)
    patch_size = image_selection_params.get('patch_size', None)

    fe_outs = []
    transforms = []
    act_hooks = []
    models = []
    clfs = []
    representations = []
    og_models = []
    for mi, param_dicts in enumerate([repr_0, repr_1]):

        if 'model' in param_dicts:
            model_name = param_dicts['model']
            ckpt_path = param_dicts.get('model_ckpt', None)
            model_out = model_loader.load_model(model_name, ckpt_path, device='cuda', eval=True)
            model = model_out['model']
            og_models.append(copy.deepcopy(model))

            transform = model_out['test_transform'] if transform_type == 'test' or patchify else model_out['transform']
            transforms.append(transform)
            fe_out = ceh.load_feature_extraction_layers(model, param_dicts['feature_extraction_params'])
            act_hook = ActivationHookV2(move_to_cpu_in_hook=move_to_cpu_in_hook, move_to_cpu_every=move_to_cpu_every)
            act_hook.register_hooks(fe_out['layer_names'], fe_out['layers'], fe_out['post_activation_func'])
            act_hooks.append(act_hook)
            fe_outs.append(fe_out)
            models.append(model)
            backbone, fc = split_model(model)
            clfs.append(fc)
            representations.append(None)

        elif 'triplet_embedding_data' in param_dicts:
            if dataset_name == 'chinese_chars' or dataset_name == 'butterflies':
                # handle the format for triplet data from humans on these datasets
                path = param_dicts['triplet_embedding_data']
                if '.pkl' in path:
                    with open(path, 'rb') as f:
                        triplet_indices = pkl.load(f)['triplets']
                else:
                    # generally used for simulated triplet data
                    triplet_indices = np.load(param_dicts['triplet_embedding_data'])['triplet']
            else:
                raise ValueError(f'Unknown dataset: {dataset_name}')

            key = list(image_group.keys())[0]
            assert triplet_indices.max() < len(image_group[key])
            fe_outs.append({'layer_names': [None]})
            transforms.append(None)
            act_hooks.append(None)
            models.append(None)
            clfs.append(None)
            og_models.append(None)

            if param_dicts['triplet_embedding_method'] == 'GNMDS':
                max_iter = 2000 if param_dicts.get('max_iter', None) is None else param_dicts['max_iter']
                max_iter = 100 if args.debug else max_iter
                estimator = GNMDS(n_components=param_dicts['num_dimensions'], verbose=True, max_iter=max_iter, )
                repr = estimator.fit_transform(triplet_indices)
                print(f'Estimator Stress: {estimator.stress_}')
                repr = torch.FloatTensor(repr)
                representations.append(repr)
            elif param_dicts['triplet_embedding_method'] == 'CKL':
                max_iter = 2000 if param_dicts.get('max_iter', None) is None else param_dicts['max_iter']
                max_iter = 100 if args.debug else max_iter
                estimator = cblearn.embedding.CKL(n_components=param_dicts['num_dimensions'], verbose=True, max_iter=max_iter, )
                repr = estimator.fit_transform(triplet_indices)
                print(f'Estimator Stress: {estimator.stress_}')
                repr = torch.FloatTensor(repr)
                representations.append(repr)
            else:
                raise ValueError(f'Unknown triplet embedding method: {param_dicts["triplet_embedding_method"]}')

        elif 'pcbm' in param_dicts:
            key = list(image_group.keys())[0]
            fe_outs.append({'layer_names': [None]})
            transforms.append(None)
            act_hooks.append(None)
            models.append(None)
            clfs.append(None)
            og_models.append(None)
            tmp = torch.FloatTensor(np.load(param_dicts['pcbm']))
            if 'exclude_dims' in param_dicts:
                mask = np.ones(tmp.shape[1], dtype=bool)
                mask[param_dicts['exclude_dims']] = False
                tmp = tmp[:, mask]

            representations.append(tmp)

    if image_selection_params.get('share_transforms'):
        transforms[0] = transforms[1]

    m0_layers = fe_outs[0]['layer_names'][::-1]
    m1_layers = fe_outs[1]['layer_names'][::-1]
    if only_last_layer:
        m0_layers = [m0_layers[0]]
        m1_layers = [m1_layers[0]]

    labels = np.array(labels)

    ci = 0
    pbar = tqdm(list(image_group.keys()))
    for image_group_key in pbar:
        pbar.set_description(f'Class {image_group_key}')

        if image_group_key not in image_group or len(image_group[image_group_key]) < 5:
            print(f'No images for class {image_group_key}')
            continue

        params = dict(
            image_list=image_group[image_group_key],
            dataset_name=dataset_name, dataset_root=dataset_root, device=device, patch_size=patch_size, labels=labels,
            patchify=patchify, fe_outs=fe_outs, transforms=transforms, act_hooks=act_hooks, models=models, raw_data=raw_data,
        )
        images_preprocessed, labels, unnormalize = shared_concept_proposals_inference(params)
        if len(images_preprocessed) == 1:
            image_samples = [unnormalize(images_preprocessed[0])] * 2
        elif len(images_preprocessed) == 2:
            image_samples = [unnormalize(images_preprocessed[0]), unnormalize(images_preprocessed[1])]
        else:
            dataset = ig_dict['dataset']
            image_samples = [np.array([os.path.join(dataset.data_root, dataset.paths[i]) for i in range(len(dataset.paths))])] * 2

        print('Comparing model concepts')
        # pl.seed_everything(42)

        for lj, m1_layer in enumerate(m1_layers): # reverse order to start from the last layer
            if m1_layer is not None:
                activations2 = act_hooks[1].layer_activations[m1_layer]
                representations[1] = activations2

            for li, m0_layer in enumerate(m0_layers):  # reverse order to start from the last layer

                if m0_layer is not None:
                    activations1 = act_hooks[0].layer_activations[m0_layer]
                    representations[0] = activations1

                if args.save_m0_representation:
                    with open(f'{output_root_folder}/m0_rep.pkl', 'wb') as f:
                        pkl.dump(representations[0], f)

                if args.save_m1_representation:
                    with open(f'{output_root_folder}/m1_rep.pkl', 'wb') as f:
                        pkl.dump(representations[1], f)

                # with open(f'{output_root_folder}/labels.pkl', 'wb') as f:
                #     pkl.dump(labels, f)

                red0 = PCA(2).fit_transform(representations[0])
                red1 = PCA(2).fit_transform(representations[1])
                repr0_mapped, repr1_mapped = None, None
                r0m_red, r1m_red = None, None
                for method_dict in config['methods']:

                    method = method_dict['method']

                    input_dict = copy.deepcopy(method_dict)
                    input_dict['output_dir'] = f'{output_root_folder}/{image_group_key}'
                    input_dict['representations'] = representations
                    input_dict['seed'] = seed
                    input_dict['red0'] = red0
                    input_dict['red1'] = red1
                    input_dict['dataset_labels'] = labels
                    input_dict['image_samples'] = image_samples[0]
                    # added so that it can be separated and added to saved outputs and can be used for viz
                    input_dict['method_dict'] = method_dict
                    method_name = method_dict.get("method_name", 'default')

                    align_reps = method_dict.get('align_representations', None)
                    if align_reps is not None and align_reps is not False:
                        align_params = method_dict.get('align_params', {})
                        if repr0_mapped is None:
                            mapping_params = {'train_fraction': align_params.get('train_fraction', 0.7),
                                              'num_steps': align_params.get('num_steps', 100),
                                              'lr': align_params.get('lr', 0.001)}
                            repr0_mapped, r0m_hist = learn_mapping(representations[0], representations[1],
                                                                   mapping_params)
                            repr1_mapped, r1m_hist = learn_mapping(representations[1], representations[0],
                                                                   mapping_params)
                            red = PCA(n_components=2)
                            r0m_red = red.fit_transform(repr0_mapped)
                            r1m_red = red.fit_transform(repr1_mapped)
                            with open(f'{output_root_folder}/r0_mapping.pkl', 'wb') as f:
                                pkl.dump(r0m_hist, f)
                            with open(f'{output_root_folder}/r1_mapping.pkl', 'wb') as f:
                                pkl.dump(r1m_hist, f)

                        input_dict['repr0_mapped'] = repr0_mapped
                        input_dict['repr1_mapped'] = repr1_mapped
                        input_dict['r0m_red'] = r0m_red
                        input_dict['r1m_red'] = r1m_red

                    # if method_name != 'rdx_nb_lb_eigc' and method_name != 'rdx_nb_lb_pagerank':
                    #     continue

                    # if method_name != 'rdx_zpls_lb_spectral':
                    #     continue

                    # if method != 'kmeans' and method != 'nmf' and method != 'pca' and method != 'cnmf':
                    #     continue
                    # if method != 'classification':
                    #     continue
                    print(method_name)
                    load_outputs = False
                    start_time = time.time()
                    if method == 'cka':
                        # continue
                        run_cka(input_dict)
                    elif method == 'classification':
                        # continue
                        run_clf(input_dict)
                    elif method == 'rdx':
                        # continue
                        # if input_dict['sim_function'] != 'zp_local_scaling':
                        #     continue
                        run_rdx(input_dict, load_outputs=load_outputs)
                    elif method == 'kmeans':
                        # continue
                        run_kmeans(input_dict, load_outputs=load_outputs)
                    elif method == 'nmf':
                        # continue
                        run_nmf(input_dict, load_outputs=load_outputs)
                    elif method == 'pca':
                        # continue
                        run_pca(input_dict, load_outputs=load_outputs)
                    elif method == 'sae':
                        # continue
                        # plt.imsave(f'{output_root_folder}/img800.png', image_samples[1][800][0])
                        run_sae(input_dict, load_outputs=load_outputs)
                    else:
                        raise ValueError(f'Unknown method: {method}')
                    end_time = time.time()

                    print(f"Time taken: {end_time - start_time}.")
            break

        for mi in range(len(fe_outs)):
            if act_hooks[mi] is not None:
                act_hooks[mi].reset_activation_dict()

        ci += 1

if __name__ == '__main__':

    main()
