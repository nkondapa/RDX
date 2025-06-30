import json
import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from matplotlib import colors
from scipy.special import comb
import itertools
from src.rdx import RDX
from src.dist_funcs import (zp_local_scaling_euclidean_distance, scale_invariant_local_biased_distance,
                            max_normalized_euclidean_distance)
import pandas as pd
import scipy as sp
import src.utils.plotting_helper as ph
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from matplotlib.markers import MarkerStyle
import matplotlib

# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']= r"\usepackage{bm} \usepackage{amsmath}"

# plt.rcParams['text.usetex'] = True  # This triggers the LaTeX compiler
# fig = plt.figure()
# fig.set_size_inches(10, 10)
# plt.title(r'\texttt{Bold Monospaced}')
# plt.plot([0, 1], [0, 1])
# plt.xlabel(r'\textbf{\texttt{RDX}}, \texttt{RDX}', fontsize=30)
# plt.ylabel(r'\texttt{RDX}')

methods = {
    'rdx_neighborhood':
        {
            '01': {'color': 'blue'},
            '10': {'color': 'brown'}
        },
    'rdx_zp_local_scaling':
        {
            '01': {'color': 'blue'},
            '10': {'color': 'brown'}
        },
    'rdx_max_normalized_distance':
        {
            '01': {'color': 'blue'},
            '10': {'color': 'brown'}
        },
    'rdx_nb_lb_spectral':
        {
            '01': {'color': 'blue'},
            '10': {'color': 'brown'}
        },
    'rdx_nb_lb_pagerank':
        {
            '01': {'color': 'blue'},
            '10': {'color': 'brown'}
        },
    'rdx_nb_lb_eigc':
        {
            '01': {'color': 'blue'},
            '10': {'color': 'brown'}
        },
    'rdx_nb_s_spectral':
        {
            '01': {'color': 'blue'},
            '10': {'color': 'brown'}
        },
    'rdx_zpls_lb_spectral':
        {
            '01': {'color': 'blue'},
            '10': {'color': 'brown'}
        },

    'rdx_zpls_s_spectral':
        {
            '01': {'color': 'blue'},
            '10': {'color': 'brown'}
        },
    'rdx_mnd_lb_spectral':
        {
            '01': {'color': 'blue'},
            '10': {'color': 'brown'}
        },
    'rdx_mnd_s_spectral':
        {
            '01': {'color': 'blue'},
            '10': {'color': 'brown'}
        },
    'nmf':
        {'0': {'color': 'green'},
         '1': {'color': 'red'}},
    'cnmf':
        {'0': {'color': 'green'},
         '1': {'color': 'red'}},
    'cnmf_ar1to0':
        {'0': {'color': 'green'},
         '1': {'color': 'red'}},
    'cnmf_ar0to1':
        {'0': {'color': 'green'},
         '1': {'color': 'red'}},
    'pca':
        {'0': {'color': 'green'},
         '1': {'color': 'red'}},
    'kmeans': {
        '0': {'color': 'green'},
        '1': {'color': 'red'}
    },
    'kmeans_ar1to0': {
        '0': {'color': 'green'},
        '1': {'color': 'red'}
    },
    'kmeans_ar0to1': {
        '0': {'color': 'green'},
        '1': {'color': 'red'}
    },
    'sae': {
        '0': {'color': 'green'},
        '1': {'color': 'red'}
    },
    'sae_ar1to0': {
        '0': {'color': 'green'},
        '1': {'color': 'red'}
    },
    'sae_ar0to1': {
        '0': {'color': 'green'},
        '1': {'color': 'red'}
    },
}

method_names = {"rdx_neighborhood": "$\mathtt{RDX}_{NB}$",
                # "rdx_zp_local_scaling": "$\mathtt{RDX}_{LS}$",
                # "rdx_max_normalized_distance": "$\mathtt{RDX}_{MN}$",
                # "rdx_nb_lb_spectral": "$\mathtt{RDX}$",
                "rdx_zp_local_scaling": "$RDX_{LS}$",
                "rdx_max_normalized_distance": "$RDX_{MN}$",
                "rdx_nb_lb_spectral": "$\mathbf{RDX}$",
                "rdx_nb_lb_eigc": "RDX_NBD_LB_EIGC",
                "rdx_nb_lb_pagerank": "$\mathtt{RDX}_{PR}$",
                "rdx_nb_s_spectral": r"$\mathtt{RDX}\ \textrm{(Sub)}$",
                "rdx_zpls_lb_spectral": "$\mathtt{RDX}_{LS}$",
                "rdx_zpls_s_spectral": r"$\mathtt{RDX}_{LS}\ \textrm{(Sub)}$",
                "rdx_mnd_lb_spectral": "$\mathtt{RDX}_{MN}$",
                "rdx_mnd_s_spectral": r"$\mathtt{RDX}_{MN}\ \textrm{(Sub)}$",
                # "rdx_nb_lb_pagerank": "$RDX_{PR}$",
                # "rdx_nb_s_spectral": "$RDX\ (Sub)$",
                # "rdx_zpls_lb_spectral": "$RDX_{LS}$",
                # "rdx_zpls_s_spectral": "$RDX_{LS}\ (Sub)$",
                # "rdx_mnd_lb_spectral": "$RDX_{MN}$",
                # "rdx_mnd_s_spectral": "$RDX_{MS}\ (Sub)$",
                "nmf": "$\mathtt{NMF}$", "cnmf": "$\mathtt{CNMF}$", "snmf": "$\mathtt{SNMF}$",
                "kmeans": "$\mathtt{KMeans}$", "pca":
                    "$\mathtt{PCA}$", "clf": "Classifier",
                "sae": "$\mathtt{SAE}$",
                "kmeans_ar1to0": "$\mathtt{KMeans}_{B'}$", "kmeans_ar0to1": "$\mathtt{KMeans}_{A'}$",
                "sae_ar1to0": "$\mathtt{SAE}_{B'}$", "sae_ar0to1": "$\mathtt{SAE}_{A'}$",
                "cnmf_ar0to1": "$\mathtt{CNMF}_{A'}$", "cnmf_ar1to0": "$\mathtt{CNMF}_{B'}$",

                # "nmf": "$NMF$", "cnmf": "$CNMF$", "snmf": "$SNMF$",
                # "kmeans": "$KMeans$", "pca":
                #     "$PCA$", "clf": "Classifier",
                # "sae": "$SAE$",
                # "kmeans_ar1to0": "$KMeans}_{B'}$", "kmeans_ar0to1": "$KMeans_{A'}$",
                # "sae_ar1to0": "$SAE_{B'}$", "sae_ar0to1": "$SAE_{A'}$",
                # "cnmf_ar0to1": "$CNMF_{A'}$", "cnmf_ar1to0": "$CNMF_{B'}$",
                "cka": "CKA"}


def process_files(params):
    exp_names = params['exp_names']
    # datasets = params['datasets']
    out_folder_name = params['out_folder_name']
    folders = []
    dataset_list = []
    output_root_folders = []
    configs = []
    for i in range(len(exp_names)):
        output_root_folder = f'{ROOT_OUTPUT_FOLDER}/{exp_names[i]}'
        configs_root_folder = f'{ROOT_CONFIG_FOLDER}/{exp_names[i]}'
        with open(f'{configs_root_folder}/configs.json', 'r') as f:
            config_paths = json.load(f)
        output_root_folders.extend([output_root_folder] * len(config_paths))
        for config_path in config_paths:
            folders.extend([config_path.split('/')[-1].split('.json')[0]])
            with open(config_path, 'r') as f:
                config = json.load(f)
                tag = "_subset_grouped"
                if config['image_selection']['dataset_name'] == 'cub_pcbm':
                    tag = ''
                dataset_list.append(config['image_selection']['dataset_name'] + tag)
                configs.append(config)

    files = params['files']
    method_dict = params['method_dict']
    method_keys = params['method_keys']
    data = {}
    cka_scores = []
    selected_indices = {}
    for method in method_keys:
        selected_indices[method] = {}
        for di in method_dict[method]:
            selected_indices[method][di] = []

    rdx_mean_cluster_affinity = {"10": [], "01": []}
    clf_scores = {"m0_acc": [], "m1_acc": [], "agreement": [], "dataset_labels": []}
    ckpt_id = []
    m0_reps = []
    m1_reps = []
    m0_mappings = []
    m1_mappings = []
    for fi, folder in enumerate(folders):
        print(fi, folder)
        ckpt_id.append(folder.split('butterflies_r18_')[-1].split('_vs')[0])
        data[folder] = {}
        dataset = dataset_list[fi]
        output_root_folder = output_root_folders[fi]
        for method, method_files in files.items():
            for method_file in method_files:
                file_path = os.path.join(output_root_folder, folder, dataset, method, method_file)
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        data[folder][(method, method_file)] = pkl.load(f)
                else:
                    if method in method_dict and method_file == 'fig_paths.pkl':
                        for di in method_dict[method]:
                            selected_indices[method][di].append(None)
                    print(f"File {file_path} not found. Skipping")
                    continue

                if method == 'cka':
                    cka_scores.append(data[folder][(method, method_file)]['cka_val'])
                    print(cka_scores[-1])
                elif method == 'clf':
                    if 'acc0' in data[folder][(method, method_file)]:
                        clf_scores["m0_acc"].append(np.mean(data[folder][(method, method_file)]['acc0']))
                        clf_scores["m1_acc"].append(np.mean(data[folder][(method, method_file)]['acc1']))
                        clf_scores["agreement"].append(np.mean(data[folder][(method, method_file)]['agreement']))
                    else:
                        clf_scores["m0_acc"].append(np.mean(data[folder][(method, method_file)]['accs']['0']))
                        clf_scores["m1_acc"].append(np.mean(data[folder][(method, method_file)]['accs']['1']))
                        clf_scores["agreement"].append(np.mean(data[folder][(method, method_file)]['agreement']['01']))

                    print(clf_scores["m0_acc"][-1], clf_scores["m1_acc"][-1], clf_scores["agreement"][-1])
                elif ((method in ['nmf', 'cnmf', 'kmeans', 'pca', 'sae',
                                  'cnmf_ar0to1', 'cnmf_ar1to0', 'kmeans_ar0to1', 'kmeans_ar1to0',
                                  'sae_ar0to1', 'sae_ar1to0']) and method_file == 'fig_paths.pkl'):
                    selected_indices[method]['0'].append(data[folder][(method, method_file)]['0']["selected_indices"])
                    selected_indices[method]['1'].append(data[folder][(method, method_file)]['1']["selected_indices"])
                elif 'rdx' in method and method_file == 'fig_paths.pkl':
                    print(method)
                    selected_indices[method]['10'].append(
                        data[folder][(method, method_file)]['10']["selected_indices"])
                    selected_indices[method]['01'].append(
                        data[folder][(method, method_file)]['01']["selected_indices"])
                    if 'rdx' in method:
                        rdx_mean_cluster_affinity["10"].append(
                            data[folder][(method, method_file)]['10']["mean_cluster_affinity"])
                        rdx_mean_cluster_affinity["01"].append(
                            data[folder][(method, method_file)]['01']["mean_cluster_affinity"])

            if method in selected_indices:
                if method in method_dict and selected_indices[method][list(method_dict[method].keys())[0]][
                    -1] is not None:
                    selected_indices[method]['null_cluster'] = data[folder][(method, 'outputs.pkl')]['inputs'].get(
                        'add_null_cluster', False)
        with open(f'{os.path.join(output_root_folder, folder)}/m0_rep.pkl', 'rb') as f:
            m0_rep = pkl.load(f)
        with open(f'{os.path.join(output_root_folder, folder)}/m1_rep.pkl', 'rb') as f:
            m1_rep = pkl.load(f)

        if os.path.exists(f'{os.path.join(output_root_folder, folder)}/r0_mapping.pkl'):
            with open(f'{os.path.join(output_root_folder, folder)}/r0_mapping.pkl', 'rb') as f:
                r0_mapping = pkl.load(f)
            with open(f'{os.path.join(output_root_folder, folder)}/r1_mapping.pkl', 'rb') as f:
                r1_mapping = pkl.load(f)
        else:
            r0_mapping = None
            r1_mapping = None
        m0_reps.append(m0_rep)
        m1_reps.append(m1_rep)
        m0_mappings.append(r0_mapping)
        m1_mappings.append(r1_mapping)

    save_folder = f'{ROOT_OUTPUT_FOLDER}/{out_folder_name}/analysis/'
    os.makedirs(save_folder, exist_ok=True)
    return dict(data=data, cka_scores=cka_scores, selected_indices=selected_indices,
                rdx_mean_cluster_affinity=rdx_mean_cluster_affinity, clf_scores=clf_scores,
                ckpt_id=ckpt_id, m0_reps=m0_reps, m1_reps=m1_reps, m0_mappings=m0_mappings,
                m1_mappings=m1_mappings), folders, save_folder


def evaluate_explanations(params):
    methods = params['method_dict']
    method_keys = params['method_keys']
    fi = params['file_index']
    repr_0 = params['repr_0'][fi]
    repr_1 = params['repr_1'][fi]
    sim_func = params['sim_func']
    sim_params = params['sim_params']
    selected_indices = params['selected_indices']

    if sim_func == 'silb':
        out = scale_invariant_local_biased_distance(repr_0, repr_1, sim_params)
    elif sim_func == 'mned':
        out = max_normalized_euclidean_distance(repr_0, repr_1, sim_params)
    elif sim_func == 'zplsed':
        out = zp_local_scaling_euclidean_distance(repr_0, repr_1, sim_params)
    else:
        raise ValueError(f"Unknown similarity function: {sim_func}")

    dist0_2d = out['dist0']
    dist1_2d = out['dist1']
    dist0 = dist0_2d.flatten()
    dist1 = dist1_2d.flatten()

    edge_sets = {}
    grid_sim = {}
    summ_scores = {}
    average_edge_sim = {"dist0_average": dist0.mean().item(), "dist1_average": dist1.mean().item()}
    for method in method_keys:
        grid_sim[method] = {}
        summ_scores[method] = {}
        for dii, di in enumerate(methods[method]):
            grid_sim[method][di] = {"s0": [], "s1": []}
            summ_scores[method][di] = {'binary': 0, 'equal': 0, 'difference': 0, 'total': 0}
            edge_set = set()
            print(method, di, params['folders'][fi])
            grids = selected_indices[method][di][fi]
            if grids is None:
                continue
            null_cluster = selected_indices[method]['null_cluster']
            if null_cluster:
                grids = grids[1:]
            [edge_set.update(list(itertools.combinations(grids[i], 2))) for i in range(len(grids))]
            edge_set_dist0 = np.array([dist0_2d[e[0], e[1]].item() for e in edge_set])
            edge_set_dist1 = np.array([dist1_2d[e[0], e[1]].item() for e in edge_set])
            edge_sets[(method, di)] = (edge_set, edge_set_dist0, edge_set_dist1)
            for i in range(len(grids)):
                if len(grids[i]) != 0:
                    mask = ~np.eye(len(grids[i]), dtype=bool).flatten()
                    grid_dist0 = dist0_2d[grids[i]][:, grids[i]].flatten()[mask]
                    grid_dist1 = dist1_2d[grids[i]][:, grids[i]].flatten()[mask]
                    grid_sim[method][di]["s0"].extend(grid_dist0.tolist())
                    grid_sim[method][di]["s1"].extend(grid_dist1.tolist())
                    if dii == 0:
                        summ_scores[method][di]['binary'] += (grid_dist0 < grid_dist1).sum().item()
                        summ_scores[method][di]['difference'] += -1 * (grid_dist0 - grid_dist1).sum().item()
                    else:
                        summ_scores[method][di]['binary'] += (grid_dist0 > grid_dist1).sum().item()
                        summ_scores[method][di]['difference'] += -1 * (grid_dist1 - grid_dist0).sum().item()
                    summ_scores[method][di]['equal'] += (grid_dist0 == grid_dist1).sum().item()
                    summ_scores[method][di]['total'] += len(grid_dist0)

    print()
    for method in method_keys:
        for di in methods[method]:
            print(method, di, summ_scores[method][di])
            if summ_scores[method][di]['total'] != 0:
                summ_scores[method][di]['binary'] = summ_scores[method][di]['binary'] / summ_scores[method][di]['total']
                summ_scores[method][di]['equal'] = summ_scores[method][di]['equal'] / summ_scores[method][di]['total']
                summ_scores[method][di]['difference'] = summ_scores[method][di]['difference'] / summ_scores[method][di][
                    'total']  # * 100

    return dict(summary=summ_scores, sim_func=sim_func, folder_name=params['folders'][fi], fi=fi,
                average_edge_sim=average_edge_sim)


def plot_explanation(params):
    eval_dicts = params['eval_dicts']
    output_root_folder = params.get('output_root_folder', None)
    methods = params['method_dict']
    method_keys = params['method_keys']
    sim_funcs = params['sim_funcs']
    if output_root_folder is not None:
        os.makedirs(output_root_folder, exist_ok=True)
    num_runs = len(eval_dicts)
    shape_map = {1: (1, 1), 3: (1, 3), 6: (2, 3), 9: (3, 3), 12: (4, 3), 15: (5, 3), 18: (6, 3)}
    fig, axes = plt.subplots(*shape_map[num_runs], figsize=(20, 20), squeeze=False)
    rank_colors = ["white"] * len(method_keys)
    rank_colors[-1] = "lightcoral"
    rank_colors[-2] = "lightgreen"
    rank_colors[-3] = "lightblue"

    x_labels = ['Method', 'B0', 'EQ0', 'D0', 'B1', 'D1', 'EQ1']
    for i, (ed_key, eval_dict) in enumerate(eval_dicts.items()):
        ax = axes.flatten()[i]
        eval_summary = eval_dict['summary']
        title = eval_dict['sim_func']
        if i % len(sim_funcs) == 0:
            title += ' | ' + str(eval_dict['folder_name'])
        # methods = params['methods']
        cell_text = []
        cell_vals = []

        for method, di_dict in eval_summary.items():
            row = [method_names[method]]
            val_row = [-1]
            for dii, dii_val in di_dict.items():
                # row = [method, dii, f"{dii_val['binary']}", f"{dii_val['difference']:0.2f}"]
                row.extend([f"{dii_val['binary']:0.2f}", f"{dii_val['equal']:0.2f}", f"{dii_val['difference']:0.2f}"])
                val_row.extend([dii_val['binary'], dii_val['equal'], dii_val['difference']])
            cell_text.append(row)
            cell_vals.append(val_row)
        cell_vals = np.array(cell_vals)
        cell_vals.argsort(axis=0)

        ranks = sp.stats.rankdata(cell_vals, axis=0, method='min') - 1
        # convert ranks to colors
        cell_colors = []
        for i in range(len(cell_text)):
            row_colors = []
            for j in range(len(cell_text[i])):
                if j == 0:
                    row_colors.append('white')
                else:
                    rank = ranks[i, j]
                    if rank < len(rank_colors):
                        row_colors.append(rank_colors[int(rank)])
                    else:
                        row_colors.append('white')
            cell_colors.append(row_colors)
        cell_colors = np.array(cell_colors)

        table = ax.table(cell_text, cell_colors, loc='center', cellLoc='center', colLabels=x_labels)
        table.set_fontsize(14)
        table.scale(1.5, 1.5)  # may help
        ax.set_title(title)
        ph.make_axes_invisible(ax)
        # if i % len(sim_funcs) == 0:
        #     ax.set_ylabel(str(eval_dict['folder_name']), fontsize=14)

    plt.tight_layout()
    if output_root_folder is not None:
        plt.savefig(f'{output_root_folder}/explanation_stats.png')
    if SHOW:
        plt.show()


def bar_plot_explanation(params):
    eval_dicts = params['eval_dicts']
    output_root_folder = params.get('output_root_folder', None)
    methods = params['method_dict']
    method_keys = params['method_keys']
    sim_funcs = params['sim_funcs']
    folders = params['folders']
    xaxis = params['xaxis']
    yaxis = params['yaxis']
    subplots_value = params['subplots']
    use_method_colors = params.get('use_method_colors', True)
    if subplots_value == 'sim_funcs':
        num_subplots = len(sim_funcs)
    else:
        raise ValueError(f"Unknown subplots value: {subplots_value}")

    if output_root_folder is not None:
        os.makedirs(output_root_folder, exist_ok=True)

    num_runs = len(eval_dicts)
    shape_map = {1: (1, 1), 3: (1, 3), 6: (2, 3), 9: (3, 3), 12: (4, 3), 15: (5, 3), 18: (6, 3)}
    rank_colors = ["white"] * len(method_keys)
    rank_colors[-1] = "lightcoral"
    rank_colors[-2] = "lightgreen"
    rank_colors[-3] = "lightblue"

    x_labels = ['Method', 'B0', 'D0', 'B1', 'D1']

    sim_func_names = {"silb": "SILB", "zplsed": "LSED", "mned": "MNED"}

    bar_groups = {}
    for i, (ed_key, eval_dict) in enumerate(eval_dicts.items()):
        exp_name, sim_func = ed_key
        ed_method_names = list(eval_dicts[ed_key]['summary'].keys())
        print()
        if subplots_value == 'sim_funcs' and xaxis == 'method':
            if sim_func not in bar_groups:
                bar_groups[sim_func] = {}
            for method in ed_method_names:
                if method not in bar_groups[sim_func]:
                    bar_groups[sim_func][method] = {'mBSR': {}, 'mTSD': {}, "BSR_0": {}, "BSR_1": {},
                                                    "TSD_0": {}, "TSD_1": {}}
                mBSR = 0
                mTSD = 0
                BSR_0 = 0
                TSD_0 = 0
                BSR_1 = 0
                TSD_1 = 0
                for dii, di in enumerate(methods[method]):
                    print(ed_key, method, di, eval_dict['summary'][method][di])
                    mBSR += eval_dict['summary'][method][di]['binary']
                    mTSD += eval_dict['summary'][method][di]['difference']
                    if dii == 0:
                        BSR_0 += eval_dict['summary'][method][di]['binary']
                        TSD_0 += eval_dict['summary'][method][di]['difference']
                    elif dii == 1:
                        BSR_1 += eval_dict['summary'][method][di]['binary']
                        TSD_1 += eval_dict['summary'][method][di]['difference']

                mBSR /= len(methods[method])
                mTSD /= len(methods[method])
                bar_groups[sim_func][method]["mBSR"][exp_name] = mBSR
                bar_groups[sim_func][method]["mTSD"][exp_name] = mTSD
                bar_groups[sim_func][method]["BSR_0"][exp_name] = BSR_0
                bar_groups[sim_func][method]["TSD_0"][exp_name] = TSD_0
                bar_groups[sim_func][method]["BSR_1"][exp_name] = BSR_1
                bar_groups[sim_func][method]["TSD_1"][exp_name] = TSD_1

    markers = ['X', 'o', 's', '^', 'D', 'v', '<', '>', '*']
    cmap = plt.get_cmap('tab20')
    method_to_color_index = {"rdx": 0,
                             "kmeans": 1,
                             "sae": 2,
                             "nmf": 3, "cnmf": 3, "pca": 4}
    fontsize = 22
    metrics = ['BSR', 'TSD']
    metrics = ['BSR']
    break_plots = True
    for mi in range(len(metrics)):
        metric = metrics[mi]
        if break_plots:
            axes = []
            figs = []
            for si in range(num_subplots):
                fig, ax = plt.subplots(1, 1, figsize=(7, 6))
                figs.append(fig)
                axes.append(ax)
            axes = np.array(axes)
        else:
            fig, axes = plt.subplots(*shape_map[num_subplots], figsize=(22, 6), squeeze=False)
            figs = [fig] * num_subplots

        for bgi, (bg_key, bar_group) in enumerate(bar_groups.items()):
            ax = axes.flatten()[bgi]
            fig = figs[bgi]
            bar_width = 0.35
            vals = []
            vals0 = []
            vals1 = []
            for i, (method, method_dict) in enumerate(bar_group.items()):
                val = np.array(list(method_dict[f'm{metric}'].values()))
                vals0.append(np.array(list(method_dict[f'{metric}_0'].values())))
                # vals0.append(np.array(list(method_dict['TSD_0'].values())))
                vals1.append(np.array(list(method_dict[f'{metric}_1'].values())))
                # vals1.append(np.array(list(method_dict['TSD_1'].values())))
                vals.append(val)
            vals = np.array(vals)
            vals0 = np.array(vals0)
            vals1 = np.array(vals1)
            pos0 = np.arange(len(bar_group.keys())) + 1 - bar_width / 2 - 0.025
            pos1 = np.arange(len(bar_group.keys())) + 1 + bar_width / 2 + 0.025
            for i in range(len(method_keys)):
                if use_method_colors:
                    ci = method_to_color_index[method_keys[i].split('_')[0]]
                else:
                    ci = i
                col = cmap(ci * 2)
                ax.boxplot(vals0.T[:, i], positions=pos0[[i]], widths=bar_width,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=col, ), medianprops=dict(color='black'),
                           whiskerprops=dict(color='black'), capprops=dict(color='black'),
                           flierprops=dict(marker='o', color=col)
                           )
                col = cmap(ci * 2 + 1)
                ax.boxplot(vals1.T[:, i], positions=pos1[[i]], widths=bar_width,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=col, color='black'), medianprops=dict(color='black'),
                           whiskerprops=dict(color='black'), capprops=dict(color='black'),
                           flierprops=dict(marker='o', color=col)
                           )

            ax.set_xticks(np.arange(len(bar_group.keys())) + 1)
            if bgi == 0 or True:
                names = [method_names[method] for method in bar_group.keys()]
                ax.set_xticklabels(names, rotation=30, fontsize=fontsize + 2, ha='right')
                if VERBOSE_LABELS:
                    ax.set_ylabel(f"{metric}", fontsize=fontsize)
                ax.set_ylim([0.0, 1.05])
                ax.set_yticks(np.arange(0.0, 1.1, 0.1))
                ax.set_yticklabels([f'{x:0.1f}' for x in np.arange(0.0, 1.1, 0.1)], fontsize=fontsize + 2)
            else:
                ax.set_yticks([])
                names = [method_names[method] for method in bar_group.keys()]
                ax.set_xticklabels(np.arange(len(bar_group.keys())) + 1, fontsize=fontsize)

            if VERBOSE_LABELS:
                ax.set_title(f'Sim. Function : {sim_func_names[bg_key]}', fontsize=24)

            fig.tight_layout()
            # fig.show()
            if output_root_folder is not None:
                fig.savefig(f'{output_root_folder}/{metric}_{sim_func_names[bg_key]}_boxplot.png', dpi=300)

        # if output_root_folder is not None:
        #     plt.savefig(f'{output_root_folder}/{metric}_boxplot.png', dpi=300)
        if SHOW:
            plt.show()


def pca_sample_plot(params):
    processed_files = params['processed_files']
    folders = params['folders']
    fi = params['file_index']
    selected_indices_dict = processed_files['selected_indices']
    nmf_type = params['nmf_type']

    m0_rep = processed_files['m0_reps'][fi]
    m0_map = processed_files['m0_mappings'][fi]
    m1_rep = processed_files['m1_reps'][fi]
    m1_map = processed_files['m1_mappings'][fi]
    method_name = 'rdx_nb_lb_spectral'
    direction = params.get('direction')
    print(folders[fi], method_name, direction)
    preds = []
    with torch.no_grad():
        if direction == '10' and m0_map is not None:
            m0_rep = m0_map['mapping_layer'](m0_rep)
            preds.append(processed_files['data'][folders[fi]][('clf', 'outputs.pkl')]['preds'][:, 1])
            preds.append(processed_files['data'][folders[fi]][('clf', 'outputs.pkl')]['preds'][:, 2])
        elif direction == '01' and m1_map is not None:
            m1_rep = m1_map['mapping_layer'](m1_rep)
            preds.append(processed_files['data'][folders[fi]][('clf', 'outputs.pkl')]['preds'][:, 0])
            preds.append(processed_files['data'][folders[fi]][('clf', 'outputs.pkl')]['preds'][:, 3])
        if direction == '10' and m0_map is None:
            preds.append(processed_files['data'][folders[fi]][('clf', 'outputs.pkl')]['preds'][:, 1])
            preds.append(processed_files['data'][folders[fi]][('clf', 'outputs.pkl')]['preds'][:, 0])
        elif direction == '01' and m1_map is None:
            preds.append(processed_files['data'][folders[fi]][('clf', 'outputs.pkl')]['preds'][:, 0])
            preds.append(processed_files['data'][folders[fi]][('clf', 'outputs.pkl')]['preds'][:, 1])
    cluster_labels = processed_files['data'][folders[fi]][(method_name, 'outputs.pkl')][
        'cluster_dict'][f'am_{direction}']['cluster_labels']
    sel_inds = selected_indices_dict[method_name][direction][fi]
    preds.append(processed_files['data'][folders[fi]][('clf', 'outputs.pkl')]['labels'])
    preds = np.stack(preds, axis=0)

    red = PCA(n_components=2)
    m0_2d = red.fit_transform(m0_rep)
    m1_2d = red.fit_transform(m1_rep)
    dataset_labels = processed_files['data'][folders[fi]][('clf', 'outputs.pkl')]['labels']
    marker_dict = dict(zip(np.unique(dataset_labels), [f'${x}$' for x in np.unique(dataset_labels)]))
    mod_tmp = plt.get_cmap('tab10')
    mod_cmap = lambda x: mod_tmp(x + 5)
    topks = [
        {"topk": selected_indices_dict[nmf_type]['0'][fi], "repr": m0_2d, "title": f"NMF(A) on Repr. A",
         "cmap": plt.get_cmap('Set1'), 'label': "NMF(A)", "markers": ['s', 'v', 'D', 'H', 'd']},
        {"topk": selected_indices_dict[nmf_type]['1'][fi], "repr": m1_2d, "title": f"NMF(B) on Repr. B",
         "cmap": mod_cmap, 'label': "NMF(B)", "markers": ['^', 'P', 'X', '*', '+']},
        {"topk": selected_indices_dict['kmeans']['0'][fi], "repr": m0_2d, "title": f"KMeans(A) on Repr. A",
         "cmap": plt.get_cmap('Set1'), 'label': "KM(A)", "markers": ['s', 'v', 'D', 'H', 'd']},
        {"topk": selected_indices_dict['kmeans']['1'][fi], "repr": m1_2d, "title": f"KMeans(B) on Repr. B",
         "cmap": mod_cmap, 'label': "KM(B)", "markers": ['^', 'P', 'X', '*', '+']},
        {"topk": selected_indices_dict['sae']['0'][fi], "repr": m0_2d, "title": f"SAE(A) on Repr. A",
         "cmap": plt.get_cmap('Set1'), 'label': "SAE(A)", "markers": ['s', 'v', 'D', 'H', 'd']},
        {"topk": selected_indices_dict['sae']['1'][fi], "repr": m1_2d, "title": f"SAE(B) on Repr. B",
         "cmap": mod_cmap, 'label': "SAE(B)", "markers": ['^', 'P', 'X', '*', '+']},
        {"topk": selected_indices_dict['rdx_nb_lb_spectral']["01"][fi][1:], "repr": m0_2d,
         "title": f"RDX(A, B) on Repr. A",
         "cmap": plt.get_cmap('Set1'), 'label': "RDX(N, B)", "markers": ['s', 'v', 'D', 'H', 'd']},
        {"topk": selected_indices_dict['rdx_nb_lb_spectral']["01"][fi][1:], "repr": m1_2d,
         "title": f"RDX(A, B) on Repr. B",
         "cmap": plt.get_cmap('Set1'), 'label': "RDX(N, B)", "markers": ['s', 'v', 'D', 'H', 'd']},
        {"topk": selected_indices_dict['rdx_nb_lb_spectral']["10"][fi][1:], "repr": m0_2d,
         "title": f"RDX(B, A) on Repr. A",
         "cmap": mod_cmap, 'label': "RDX(B, A)", "markers": ['^', 'P', 'X', '*', '+']},
        {"topk": selected_indices_dict['rdx_nb_lb_spectral']["10"][fi][1:], "repr": m1_2d,
         "title": f"RDX(B, A) on Repr. B",
         "cmap": mod_cmap, 'label': "RDX(B, A)", "markers": ['^', 'P', 'X', '*', '+']},
    ]
    fontsize = 30
    class_markers = ['s', 'v', 'D', '^', 'P', 'X', 'H', 'd', 'p', 'x', 'h', '8', '*', '+', '1', '2', '3', ]
    order = [0, 2, 4, 1, 3, 5]
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    if VERBOSE_LABELS:
        # save all plots in a single figure for quick analysis
        fig, axes = plt.subplots(5, 2, figsize=(12, 24))
    else:
        # save individual plots for making a paper quality figure
        figs = []
        axes = []
        for i in range(10):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            figs.append(fig)
            axes.append(ax)

    axes = np.array(axes)
    tmp = plt.get_cmap('Set2')
    base_cmap = lambda x: tmp(x + 0)
    # base_cmap = plt.get_cmap('Dark2')
    for axi, ax in enumerate(axes.flatten()):
        # ax.scatter(x0[:, 0], x0[:, 1], color='gray', alpha=0.5)
        # axi = order[axi]
        topk_d = topks[axi]
        topk_indices = topk_d["topk"]
        x = topk_d["repr"]
        cmap = topk_d["cmap"]
        num_c = len(topk_indices)
        marker_list = topk_d["markers"]
        all_inds = []
        for ti in topk_indices:
            all_inds.extend(ti)
        all_inds = torch.unique(torch.tensor(all_inds))

        for li, lab_i in enumerate(np.unique(dataset_labels)):
            not_sel_inds = torch.zeros(len(x), dtype=torch.bool)
            label_inds = np.where(dataset_labels == lab_i)[0]
            # label_inds = np.random.choice(label_inds, 100, replace=False)
            not_sel_inds[label_inds] = True
            not_sel_inds[all_inds] = False
            # label = f"Class {i}" if axi == 0 else None
            label = None
            ax.scatter(x[not_sel_inds, 0], x[not_sel_inds, 1], marker=None, color=base_cmap(li), alpha=0.1, s=100,
                       label=label)

        for i in range(num_c):
            lab_i = np.unique(dataset_labels)[i]
            sel_inds = topk_indices[i]

            # ax.scatter(x[not_sel_inds, 0], x[not_sel_inds, 1], color=cmap(i), alpha=0.1)
            # dl_sel_list = dataset_labels[sel_inds]
            for si in sel_inds:
                dsi = dataset_labels[si]
                ax.scatter(x[si, 0], x[si, 1], color=cmap(i), marker=marker_list[i], s=300, zorder=10, alpha=1)

            # hidden label
            label = f'{topk_d["label"]} {i}'
            ax.scatter([0], [0], color=cmap(i), marker=marker_list[i], alpha=0, s=100, zorder=10, label=label)

            # sel_inds = topk_indices[:, i]
            # ax.scatter(x0[sel_inds, 0], x0[sel_inds, 1], color=cmap(i))

        if VERBOSE_LABELS:
            ax.set_title(topk_d["title"], fontsize=fontsize)

        ax.set_xlim([min(m0_2d[:, 0].min(), m1_2d[:, 0].min()) - 0.1, max(m0_2d[:, 0].max(), m1_2d[:, 0].max()) + 1])
        ax.set_ylim([min(m0_2d[:, 1].min(), m1_2d[:, 1].min()) - 0.1, max(m0_2d[:, 1].max(), m1_2d[:, 1].max()) + 1])
        ax.set_xticks([])
        ax.set_yticks([])

        # leg = ax.legend(loc='upper right', fontsize=20)
        # for lh in leg.legendHandles:
        #     lh.set_alpha(1)

    if VERBOSE_LABELS:
        axes[0, 0].set_ylabel("Top-10 Samples per Concept", fontsize=16)
        axes[0, 0].set_ylabel("Repr. A", fontsize=fontsize)
        axes[1, 0].set_ylabel("Repr. B", fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(f'{params["output_root_folder"]}/pca_samples_plot.{EXT}')
    else:
        for i, fig in enumerate(figs):
            title = topks[i]['title'].replace(' ', '_').replace('.', '').lower()
            fig.tight_layout()
            fig.savefig(f'{params["output_root_folder"]}/pca_samples_plot_{title}.{EXT}')

    if SHOW:
        plt.show()
    plt.close('all')


def coeff_matrix_plot(params):
    processed_files = params['processed_files']
    folders = params['folders']
    fi = params['file_index']
    fontsize = 30
    fig, axes = plt.subplots(1, 4, squeeze=False)
    fig.set_size_inches(22, 6)
    nd, nc = processed_files['data'][folders[fi]][('nmf', 'outputs.pkl')]['repr_0']['U'].shape
    axes[0, 0].imshow(processed_files['data'][folders[fi]][('nmf', 'outputs.pkl')]['repr_0']['U'], aspect='auto',
                      interpolation='nearest')
    axes[0, 1].imshow(processed_files['data'][folders[fi]][('nmf', 'outputs.pkl')]['repr_1']['U'], aspect='auto',
                      interpolation='nearest')
    axes[0, 2].imshow(processed_files['data'][folders[fi]][('sae', 'outputs.pkl')]['repr_0']['U'], aspect='auto',
                      interpolation='nearest')
    axes[0, 3].imshow(processed_files['data'][folders[fi]][('sae', 'outputs.pkl')]['repr_1']['U'], aspect='auto',
                      interpolation='nearest')
    for axi, ax in enumerate(axes.flatten()):
        if axi == 0:
            ax.set_yticks(np.arange(0, nd, 200))
            ax.set_yticklabels(np.arange(0, nd, 200), fontsize=fontsize)
        else:
            ax.set_yticks([])
        ax.set_xticks(np.arange(nc))
        ax.set_xticklabels(np.arange(nc), fontsize=fontsize)

    print(processed_files['data'][folders[fi]][('sae', 'outputs.pkl')]['repr_1']['U'][800])
    plt.tight_layout()
    plt.savefig(f'{params["output_root_folder"]}/{folders[fi]}_coeffs.png', dpi=300)
    if SHOW:
        plt.show()


def intermediate_steps_plots(params, part1=True, part2=True):
    processed_files = params['processed_files']
    folders = params['folders']
    file_index = params['file_index']
    output_root_folder = params['output_root_folder']

    # subset_inds = np.sort(np.random.choice(np.arange(0, 1500), 50, replace=False))
    r0_dm = processed_files['data'][folders[file_index]][
        ('rdx_nb_lb_spectral', 'outputs.pkl')]['graph_dict']['r0_dm']
    r1_dm = processed_files['data'][folders[file_index]][
        ('rdx_nb_lb_spectral', 'outputs.pkl')]['graph_dict']['r1_dm']
    subset_inds = np.arange(0, r0_dm.shape[0])

    r0_dm = r0_dm.cpu().numpy()[subset_inds][:, subset_inds]
    r1_dm = r1_dm.cpu().numpy()[subset_inds][:, subset_inds]

    if part1:
        fig, axes = plt.subplots(1, 1)
        axes.imshow(r0_dm, aspect='auto', interpolation='nearest')
        if VERBOSE_LABELS:
            axes.set_title(f"RA DM")
            axes.set_xlabel("Samples")
            axes.set_ylabel("Samples")
        axes.tick_params(axis='both', which='major', labelsize=18)
        plt.tight_layout()
        plt.savefig(f'{output_root_folder}/{folders[file_index]}_r0_dm.{EXT}', dpi=300)

        fig, axes = plt.subplots(1, 1)
        axes.imshow(r1_dm, aspect='auto', interpolation='nearest')
        if VERBOSE_LABELS:
            axes.set_title(f"RB DM")
            axes.set_xlabel("Samples")
            axes.set_ylabel("Samples")
        # set ticks fontsize
        axes.tick_params(axis='both', which='major', labelsize=18)
        plt.tight_layout()
        plt.savefig(f'{output_root_folder}/{folders[file_index]}_r1_dm.{EXT}', dpi=300)
        if SHOW:
            plt.show()

    diff_10 = processed_files['data'][folders[file_index]][
                  ('rdx_nb_lb_spectral', 'outputs.pkl')]['graph_dict']['diff_10'][subset_inds][:, subset_inds]
    diff_01 = processed_files['data'][folders[file_index]][
                  ('rdx_nb_lb_spectral', 'outputs.pkl')]['graph_dict']['diff_01'][subset_inds][:, subset_inds]
    am_10 = processed_files['data'][folders[file_index]][
                ('rdx_nb_lb_spectral', 'outputs.pkl')]['graph_dict']['diff_10'][subset_inds][:, subset_inds]
    am_01 = processed_files['data'][folders[file_index]][
                ('rdx_nb_lb_spectral', 'outputs.pkl')]['graph_dict']['diff_01'][subset_inds][:, subset_inds]

    if part1:
        fig, axes = plt.subplots(1, 1)
        im = axes.imshow(diff_01, aspect='auto', interpolation='nearest', cmap='bwr')
        axes.tick_params(axis='both', which='major', labelsize=18)
        fig.colorbar(im, ax=axes)
        if VERBOSE_LABELS:
            axes.set_title(f"RA DM")
            axes.set_xlabel("Samples")
            axes.set_ylabel("Samples")
        plt.tight_layout()
        plt.savefig(f'{output_root_folder}/{folders[file_index]}_diff_01.{EXT}', dpi=300)
        fig, axes = plt.subplots(1, 1)
        im = axes.imshow(diff_10, aspect='auto', interpolation='nearest', cmap='bwr')
        axes.tick_params(axis='both', which='major', labelsize=18)
        fig.colorbar(im, ax=axes)
        if VERBOSE_LABELS:
            axes.set_title(f"RB DM")
            axes.set_xlabel("Samples")
            axes.set_ylabel("Samples")
        plt.tight_layout()
        plt.savefig(f'{output_root_folder}/{folders[file_index]}_diff_10.{EXT}', dpi=300)
        if SHOW:
            plt.show()

    inds = []
    breaks = []
    # cl_dists = {}
    cl_affins = {}
    cl_diffs = {}
    cl_inds = {}
    direction = params['direction']
    if direction == '10':
        diff_mat = diff_10
        affinity_mat = am_10
        cl_labels = processed_files['data'][folders[file_index]][
            ('rdx_nb_lb_spectral', 'outputs.pkl')]['cluster_dict']['am_10']['cluster_labels']
        title = 'Clustered Diff. Matrix (B - A)'
    elif direction == '01':
        diff_mat = diff_01
        affinity_mat = am_01
        cl_labels = processed_files['data'][folders[file_index]][
            ('rdx_nb_lb_spectral', 'outputs.pkl')]['cluster_dict']['am_01']['cluster_labels']
        title = 'Clustered Diff. Matrix (A - B)'
    else:
        raise ValueError(f"Unknown direction: {direction}")

    if part2:
        fig, axes = plt.subplots(1, 1)
        for cli in np.unique(cl_labels):
            _cl_inds = np.where(cl_labels == cli)[0]
            inds.extend(_cl_inds)
            cl_inds[cli] = _cl_inds
            breaks.append(len(inds))
            cl_diffs[cli] = diff_mat[_cl_inds][:, _cl_inds]
            cl_affins[cli] = affinity_mat[_cl_inds][:, _cl_inds]
            # cl_dists[cli] = rn_dist_mat[_cl_inds][:, _cl_inds]

        inds = np.array(inds)
        diff_mat_min, diff_mat_max = diff_mat.min(), diff_mat.max()
        im = axes.imshow(diff_mat[inds][:, inds], cmap='bwr', vmin=diff_mat_min, vmax=diff_mat_max)
        # fig.colorbar(im, ax=axes[0, 2])

        # fig, axes = plt.subplots(1, 1)
        for bi, b in enumerate(breaks):
            b = b - 0.5
            prev = breaks[bi - 1] if bi > 0 else 0
            for ax in [axes]:
                # ax.plot([10, 10], [10, breaks[0]], color='k', zorder=10)
                ax.plot([prev, b], [prev, prev], color='k')
                ax.plot([prev, b], [b, b], color='k')
                ax.plot([prev, prev], [prev, b], color='k')
                ax.plot([b, b], [prev, b], color='k')
            # axes[1, 1].plot([prev, b], [prev, prev], color='m')
            # axes[1, 1].plot([prev, b], [b, b], color='m')
            # axes[1, 1].plot([prev, prev], [prev, b], color='m')
            # axes[1, 1].plot([b, b], [prev, b], color='m')

        if VERBOSE_LABELS:
            axes.set_title(title, fontsize=20)
            axes.set_xlabel("Sorted Samples", fontsize=20)
            axes.set_ylabel("Sorted Samples", fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{output_root_folder}/{folders[file_index]}_spectral_clusters_{direction}.{EXT}', dpi=300)
        if SHOW:
            plt.show()
        print()


def main(exp_names, datasets, main_params):
    files = main_params['files']
    datasets = None

    method_keys = list(files.keys())
    method_keys = method_keys[2:]
    params = dict(
        exp_names=exp_names,
        datasets=datasets,
        out_folder_name=main_params.get('out_folder_name', exp_names[0]),
        files=files,
        method_dict=methods,
        method_keys=method_keys
    )
    processed_files, folders, output_root_folder = process_files(params)

    pca_plot_params = main_params.get('pca_plot_params', None)
    if pca_plot_params is not None:
        file_indices = pca_plot_params.get('file_indices', None)
        for file_index in file_indices:
            for di, direction in enumerate(pca_plot_params['directions']):
                plot_params = dict(processed_files=processed_files, folders=folders,
                                   file_index=file_index, nmf_type=pca_plot_params.get('nmf_type', 'nmf'),
                                   output_root_folder=output_root_folder, direction=direction)

                if main_params.get('plot_intermediates', False):
                    intermediate_steps_plots(plot_params, part1=di==0, part2=True)

                if processed_files['m0_mappings'][file_index] is not None or di == 0:
                    pca_sample_plot(plot_params)

    if main_params.get("plot_bsr", False):
        sim_funcs = ['silb', 'zplsed', 'mned']
        sim_params = [
            {'beta': 5, 'gamma': 0.05},
            {'scaling_neighbor': 7, 'beta': 1},
            {'scaling_neighbor': 7, 'beta': 1, 'gamma': 0.05}
        ]
        eval_dicts = {}
        for fi in range(len(folders)):
            for si, sim_func in enumerate(sim_funcs):
                params = dict(
                    file_index=fi,
                    repr_0=processed_files['m0_reps'],
                    repr_1=processed_files['m1_reps'],
                    sim_func=sim_func,
                    sim_params=sim_params[si],
                    selected_indices=processed_files['selected_indices'],
                    method_dict=methods,
                    method_keys=method_keys,
                    folders=folders,
                )
                eval_dicts[(folders[fi], sim_func)] = evaluate_explanations(params)

        plot_params = dict(eval_dicts=eval_dicts, sim_funcs=sim_funcs, method_dict=methods,
                           method_keys=method_keys,
                           output_root_folder=output_root_folder)
        # plot_explanation(plot_params)
        plot_params = dict(eval_dicts=eval_dicts, sim_funcs=sim_funcs, method_dict=methods,
                           method_keys=method_keys, folders=folders,
                           output_root_folder=output_root_folder, xaxis='method', yaxis='binary_average',
                           subplots='sim_funcs')
        bar_plot_params = main_params.get('bar_plot_params', {})
        bar_plot_params.update(plot_params)
        bar_plot_explanation(bar_plot_params)

    if main_params.get("summarize_clusters", False):
        output_path = f'{output_root_folder}/cluster_meta_summaries/'
        os.makedirs(output_path, exist_ok=True)
        sizes = [10, 25, 50, 50]
        sizes = [(50, 30)] * 10
        for fi in range(len(folders)):
            fig, axes = plt.subplots(len(method_keys), 2, figsize=(sizes[fi][0], sizes[fi][1]))
            for mi, method in enumerate(method_keys):
                for dii, di in enumerate(methods[method]):
                    path = processed_files['data'][folders[fi]][(method, 'fig_paths.pkl')][di]['cluster_summ']
                    img = Image.open(path)
                    axes[mi, dii].imshow(img)
                    axes[mi, dii].axis('off')
                    axes[mi, dii].set_title(f"{method} {di}")
            plt.tight_layout()
            plt.savefig(f'{output_path}/{folders[fi]}_cms.png')

    plt.close('all')


################################################# RUN FUNCTIONS ########################################################


def analyze_mnist_835():
    exp_names = [
        "mnist_835_experiment"
    ]
    files = {
        'cka': ['outputs.pkl'],
        'clf': ['outputs.pkl'],
        'rdx_nb_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],

        # 'rdx_nb_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_zpls_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_mnd_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        'kmeans': ['outputs.pkl', 'fig_paths.pkl'],
        'sae': ['outputs.pkl', 'fig_paths.pkl'],
        'nmf': ['outputs.pkl', 'fig_paths.pkl'],
    }
    datasets = ['mnist_835_subset_grouped']
    # for exp_name in exp_names:
    main(exp_names, datasets, main_params={"files": files,
                                           "pca_plot_params": {"directions": ["01", "10"],
                                                               "file_indices": [0]},
                                           "plot_intermediates": True,
                                           "plot_bsr": True,
                                           "summarize_clusters": True,
                                           })

def analyze_mnist_modification_experiments():
    exp_names = [
        "mnist_modification_experiments_k=3"
    ]
    files = {
        'cka': ['outputs.pkl'],
        'clf': ['outputs.pkl'],
        'rdx_nb_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_zpls_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_mnd_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],

        # 'rdx_nb_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_zpls_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_mnd_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        'kmeans': ['outputs.pkl', 'fig_paths.pkl'],
        'sae': ['outputs.pkl', 'fig_paths.pkl'],
        'nmf': ['outputs.pkl', 'fig_paths.pkl'],
        'pca': ['outputs.pkl', 'fig_paths.pkl'],
    }
    datasets = ['mnist_subset_grouped']
    main(exp_names, datasets, main_params={"files": files,
                                           "pca_plot_params": {"directions": ["01", "10"],
                                                               "file_indices": [0]},
                                           "plot_intermediates": True,
                                           "plot_bsr": True,
                                           "summarize_clusters": False,
                                           })

def analyze_cub_pcbm_experiments():
    exp_names = [
        'cub_pcbm_v_cub_masked_pcbm',
    ]
    files = {
        'cka': ['outputs.pkl'],
        'clf': ['outputs.pkl'],
        'rdx_nb_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_zpls_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_mnd_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],

        # 'rdx_nb_lb_eigc': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_nb_lb_pagerank': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_nb_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_zpls_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_mnd_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        'kmeans': ['outputs.pkl', 'fig_paths.pkl'],
        'sae': ['outputs.pkl', 'fig_paths.pkl'],
        'cnmf': ['outputs.pkl', 'fig_paths.pkl'],
        'pca': ['outputs.pkl', 'fig_paths.pkl'],
    }
    datasets = ['cub_pcbm']
    main(exp_names, datasets, main_params={"files": files,
                                           "pca_plot_params": {"directions": ["01", "10"],
                                                               "nmf_type": "cnmf",
                                                               "file_indices": [0]},
                                           "plot_intermediates": False,
                                           "plot_bsr": True,
                                           "summarize_clusters": True,
                                           })


def analyze_unaligned_real_model_experiments():
    exp_names = [
        "dino_vs_dinov2_imagenet",
        "clip_vs_clipinat_inat"
    ]
    files = {
        'cka': ['outputs.pkl'],
        'clf': ['outputs.pkl'],
        'rdx_nb_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_zpls_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_mnd_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_nb_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_zpls_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_mnd_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        'kmeans': ['outputs.pkl', 'fig_paths.pkl'],
        'sae': ['outputs.pkl', 'fig_paths.pkl'],
        'cnmf': ['outputs.pkl', 'fig_paths.pkl'],
        'pca': ['outputs.pkl', 'fig_paths.pkl'],
    }
    datasets = [
        # "cub_pcbm",
        'imagenet_subset_grouped', 'inatdl_subset_grouped'
    ]
    # for exp_name in exp_names:
    main(exp_names, datasets, main_params={"files": files, 'out_folder_name': 'real',
                                           "pca_plot_params": {"directions": ["01", "10"],
                                                               "file_indices": list(range(7))
                                                               },
                                           "plot_intermediates": True,
                                           "plot_bsr": True,
                                           "summarize_clusters": True,
                                           })


def analyze_aligned_real_model_experiments():
    exp_names = [
        "dino_vs_dinov2_imagenet_ar",
        "clip_vs_clipinat_inat_ar"
    ]
    files = {
        'cka': ['outputs.pkl'],
        'clf': ['outputs.pkl'],
        'rdx_nb_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_zpls_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_mnd_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],

        # 'rdx_nb_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_zpls_lb_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_zpls_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        # 'rdx_mnd_s_spectral': ['outputs.pkl', 'fig_paths.pkl'],
        'kmeans_ar0to1': ['outputs.pkl', 'fig_paths.pkl'],
        'kmeans_ar1to0': ['outputs.pkl', 'fig_paths.pkl'],
        'sae_ar0to1': ['outputs.pkl', 'fig_paths.pkl'],
        'sae_ar1to0': ['outputs.pkl', 'fig_paths.pkl'],
        'cnmf_ar0to1': ['outputs.pkl', 'fig_paths.pkl'],
        'cnmf_ar1to0': ['outputs.pkl', 'fig_paths.pkl'],
    }
    datasets = [
        # "cub_pcbm",
        'imagenet_subset_grouped', 'inatdl_subset_grouped'
    ]

    main(exp_names, datasets, main_params={"files": files,
                                           "pca_plot_params": {"directions": ["01", "10"],
                                                               "file_indices": list(range(7))
                                                               },
                                           "plot_intermediates": True,
                                           "plot_bsr": True,
                                           "summarize_clusters": True,
                                           })

if __name__ == "__main__":
    global ROOT_OUTPUT_FOLDER
    global ROOT_CONFIG_FOLDER
    global SHOW

    global EXT
    global VERBOSE_LABELS
    save_setting = 'analysis'
    if save_setting == 'paper':
        EXT = 'pdf'
        VERBOSE_LABELS = False
    elif save_setting == 'analysis':
        EXT = 'png'
        VERBOSE_LABELS = True

    ROOT_OUTPUT_FOLDER = "./outputs"
    ROOT_CONFIG_FOLDER = "./comparison_configs"
    SHOW = True
    # analyze_mnist_835()
    analyze_mnist_modification_experiments()
    # analyze_cub_pcbm_experiments()

    # @TODO: Test these
    # analyze_unaligned_real_model_experiments()
    # analyze_aligned_real_model_experiments()
    exit()
