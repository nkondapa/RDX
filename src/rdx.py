import torch
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, AffinityPropagation
from scipy.sparse import linalg as sparse_linalg
import scipy as sp
import os
from src.utils import plotting_helper as ph
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image
import torchvision
from src.cka import CudaCKA

class RDX:

    def __init__(self):
        pass

    def fit(self, params):
        representations = params['representations']
        repr0 = representations[0]
        repr1 = representations[1]

        if params.get('align_representations', False):
            repr0_mapped = params['repr0_mapped']
            repr1_mapped = params['repr1_mapped']

            graph_dict_a = self.construct_graph(repr0_mapped, repr1, params)
            graph_dict_b = self.construct_graph(repr0, repr1_mapped, params)
            # merge the two graph dicts
            graph_dict = {}
            graph_dict['am_10'] = graph_dict_a['am_10']
            graph_dict['dm_10'] = graph_dict_a['dm_10']
            graph_dict['diff_10'] = graph_dict_a['diff_10']

            graph_dict['am_01'] = graph_dict_b['am_01']
            graph_dict['dm_01'] = graph_dict_b['dm_01']
            graph_dict['diff_01'] = graph_dict_b['diff_01']

            graph_dict['r0_dm'] = graph_dict_b['r0_dm']
            graph_dict['r0_am'] = graph_dict_b['r0_am']

            graph_dict['r1_dm'] = graph_dict_a['r1_dm']
            graph_dict['r1_am'] = graph_dict_a['r1_am']

        else:
            graph_dict = self.construct_graph(repr0, repr1, params)

        cluster_dict = self.cluster_graph(graph_dict, params)

        output_dict = dict(graph_dict=graph_dict, cluster_dict=cluster_dict, )
        return output_dict

    @staticmethod
    def construct_graph(repr0, repr1, params):
        sim_function = params['sim_function']
        if sim_function == 'neighborhood':
            beta = params.get('beta', 5)
            diff_function = params.get('diff_function', 'locally_biased')
            normalize_diff_mat_by_abs_max = params.get('normalize_diff_mat_by_abs_max', False)
            num_samples = repr0.shape[0]
            gamma = params.get('gamma', None)
            if gamma is None:
                gamma_scale = params.get('gamma_scale', 80)
                gamma = gamma_scale / num_samples

            print(f"Beta: {beta}, Gamma: {gamma}")

            repr0 = torch.FloatTensor(repr0)
            repr1 = torch.FloatTensor(repr1)

            r0_dm = torch.cdist(repr0, repr0)
            r1_dm = torch.cdist(repr1, repr1)

            r0_sort = r0_dm.argsort()
            r0_dm = torch.argsort(r0_sort).float()
            r1_sort = r1_dm.argsort()
            r1_dm = torch.argsort(r1_sort).float()

            r1_am = torch.exp(-beta * r1_dm)
            r0_am = torch.exp(-beta * r0_dm)

            if params['guidance'] is not None:
                if params['guidance'] == 'classifier':
                    guid_labels = params['preds']
                    print('Using classifier guidance')
                elif params['guidance'] == 'ground_truth':
                    guid_labels = [params['dataset_labels'], params['dataset_labels']]
                    print('Using ground truth guidance')
                else:
                    raise ValueError(f"Unknown guidance {params['guidance']}")

                dms = [r0_dm, r1_dm]
                for i in range(2):
                    _p = guid_labels[i]
                    null_val = dms[i].max() * 5
                    for pi in np.unique(_p):
                        mask = _p == pi
                        mask = torch.BoolTensor(mask)
                        not_mask = ~mask
                        print(dms[i][mask][:, not_mask].shape)
                        mask_idx = torch.where(mask)[0]
                        not_mask_idx = torch.where(not_mask)[0]

                        dms[i][mask_idx[:, None], not_mask_idx] = null_val
                        dms[i][not_mask_idx[:, None], mask_idx] = null_val

            if diff_function == 'locally_biased':
                # needed because dm can be zero for mnd
                denom = torch.min(torch.stack([r1_dm, r0_dm]), dim=0)[0] + 1

                diff_10 = torch.tanh(gamma * (r1_dm - r0_dm) / denom)
                diff_01 = torch.tanh(gamma * (r0_dm - r1_dm) / denom)
            elif diff_function == 'standard':
                diff_10 = r1_dm - r0_dm
                diff_01 = r0_dm - r1_dm
            else:
                raise ValueError(f"Unknown diff function {diff_function}")
            # fig, axes = plt.subplots(1, 2)
            # axes[0].imshow(diff_10)
            # axes[1].imshow(diff_01)
            # plt.show()
            #
            # fig, axes = plt.subplots(1, 2)
            # random_inds = np.random.choice(np.arange(diff_10.shape[0] * diff_10.shape[1]), 1000, replace=False)
            # axes[0].hist(diff_10.flatten()[random_inds].numpy())
            # axes[1].hist(diff_01.flatten()[random_inds].numpy())
            # plt.show()
            # diff_10 = torch.tanh(gamma * (r1_dm - r0_dm) / (r1_dm))
            # diff_01 = torch.tanh(gamma * (r0_dm - r1_dm) / (r0_dm))
            if normalize_diff_mat_by_abs_max:
                diff_10 = diff_10 / diff_10.abs().max()
                diff_01 = diff_01 / diff_01.abs().max()

            am_10 = dm_01 = torch.exp(-beta * diff_10)
            am_01 = dm_10 = torch.exp(-beta * diff_01)

            # affinity_mat = am_10
            # r1_red = PCA(2).fit_transform(student_repr.detach().cpu().numpy())
            # r2_red = PCA(2).fit_transform(teacher_repr.detach().cpu().numpy())

            return dict(am_10=am_10, am_01=am_01, dm_10=dm_10, dm_01=dm_01, r0_am=r0_am, r1_am=r1_am, r0_dm=r0_dm, r1_dm=r1_dm,
                        diff_10=diff_10, diff_01=diff_01, r0_sort=r0_sort, r1_sort=r1_sort)

        elif sim_function == 'max_normalized_distance':
            beta = params.get('beta', 5)
            diff_function = params.get('diff_function', 'locally_biased')
            normalize_diff_mat_by_abs_max = params.get('normalize_diff_mat_by_abs_max', False)
            num_samples = repr0.shape[0]
            gamma = params.get('gamma', None)
            if gamma is None:
                gamma_scale = params.get('gamma_scale', 80)
                gamma = gamma_scale / num_samples
            repr0 = torch.FloatTensor(repr0)
            repr1 = torch.FloatTensor(repr1)

            r0_dm = torch.cdist(repr0, repr0)
            r1_dm = torch.cdist(repr1, repr1)

            r0_dm = r0_dm / r0_dm.max()
            r1_dm = r1_dm / r1_dm.max()

            r1_am = torch.exp(-beta * r1_dm)
            r0_am = torch.exp(-beta * r0_dm)

            diff_10 = r1_dm - r0_dm
            diff_01 = r0_dm - r1_dm

            if diff_function == 'locally_biased':
                # needed because dm can be zero for mnd
                denom = torch.min(torch.stack([r1_dm, r0_dm]), dim=0)[0] + 1
                diff_10 = torch.tanh(gamma * (r1_dm - r0_dm) / denom)
                diff_01 = torch.tanh(gamma * (r0_dm - r1_dm) / denom)
            elif diff_function == 'standard':
                diff_10 = r1_dm - r0_dm
                diff_01 = r0_dm - r1_dm
            else:
                raise ValueError(f"Unknown diff function {diff_function}")
            # diff_10 = torch.tanh(gamma * (r1_dm - r0_dm) / (r1_dm))
            # diff_01 = torch.tanh(gamma * (r0_dm - r1_dm) / (r0_dm))
            if normalize_diff_mat_by_abs_max:
                diff_10 = diff_10 / diff_10.abs().max()
                diff_01 = diff_01 / diff_01.abs().max()

            am_10 = dm_01 = torch.exp(-beta * diff_10)
            am_01 = dm_10 = torch.exp(-beta * diff_01)

            if params['classifier_guided']:
                preds = params['preds']
                ams = [r0_am, r1_am]
                for i in range(2):
                    _p = preds[i]
                    for pi in np.unique(_p):
                        mask = _p == pi
                        mask = torch.BoolTensor(mask)
                        not_mask = ~mask
                        print(ams[i][mask][:, not_mask].shape)
                        mask_idx = torch.where(mask)[0]
                        not_mask_idx = torch.where(not_mask)[0]

                        ams[i][mask_idx[:, None], not_mask_idx] = 0
                        ams[i][not_mask_idx[:, None], mask_idx] = 0

            # affinity_mat = am_10
            # r1_red = PCA(2).fit_transform(student_repr.detach().cpu().numpy())
            # r2_red = PCA(2).fit_transform(teacher_repr.detach().cpu().numpy())

            return dict(am_10=am_10, am_01=am_01, dm_10=dm_10, dm_01=dm_01, r0_am=r0_am, r1_am=r1_am, r0_dm=r0_dm, r1_dm=r1_dm,
                        diff_10=diff_10, diff_01=diff_01)

        elif sim_function == 'zp_local_scaling':
            sigma = params.get('sigma', 7)
            beta = params.get('beta', 5)
            diff_function = params.get('diff_function', 'locally_biased')
            num_samples = repr0.shape[0]
            gamma = params.get('gamma', None)
            if gamma is None:
                gamma_scale = params.get('gamma_scale', 80)
                gamma = gamma_scale / num_samples
            normalize_diff_mat_by_abs_max = params.get('normalize_diff_mat_by_abs_max', False)
            print(f"Beta: {beta}, Sigma: {sigma}")
            repr0 = torch.FloatTensor(repr0)
            repr1 = torch.FloatTensor(repr1)
            dist0 = torch.cdist(repr0, repr0, p=2)
            dist1 = torch.cdist(repr1, repr1, p=2)
            r0_sort = dist0.argsort()
            # r0_nn = torch.argsort(r0_sort)
            r1_sort = dist1.argsort()
            # r1_nn = torch.argsort(r1_sort)

            ls0 = torch.gather(dist0, 1, r0_sort[:, None, sigma])
            ls1 = torch.gather(dist1, 1, r1_sort[:, None, sigma])

            ls0_denom = ls0 @ ls0.T
            ls1_denom = ls1 @ ls1.T

            # fig, axes = plt.subplots(1, 3)
            # axes[0].imshow(dist0 @ dist0.T)
            # axes[1].imshow(dist0 ** 2)
            # axes[2].imshow(dist0 @ dist0.T - dist0 ** 2)
            # plt.show()
            r0_dm = (dist0 * dist0.T) / ls0_denom
            r1_dm = (dist1 * dist0.T) / ls1_denom
            # r0_dm = r0_dm / r0_dm.max()
            # r1_dm = r1_dm / r1_dm.max()
            r0_am = torch.exp(-beta * r0_dm)
            r1_am = torch.exp(-beta * r1_dm)

            if diff_function == 'locally_biased':
                # needed because dm can be zero for mnd
                denom = torch.min(torch.stack([r1_dm, r0_dm]), dim=0)[0] + 1
                diff_10 = torch.tanh(gamma * (r1_dm - r0_dm) / denom)
                diff_01 = torch.tanh(gamma * (r0_dm - r1_dm) / denom)
            elif diff_function == 'standard':
                diff_10 = r1_dm - r0_dm
                diff_01 = r0_dm - r1_dm
            else:
                raise ValueError(f"Unknown diff function {diff_function}")

            if normalize_diff_mat_by_abs_max:
                diff_10 = diff_10 / diff_10.abs().max()
                diff_01 = diff_01 / diff_01.abs().max()

            am_10 = dm_01 = torch.exp(-beta * diff_10)
            am_01 = dm_10 = torch.exp(-beta * diff_01)

            if params['classifier_guided']:
                preds = params['preds']
                ams = [r0_am, r1_am]
                for i in range(2):
                    _p = preds[i]
                    for pi in np.unique(_p):
                        mask = _p == pi
                        mask = torch.BoolTensor(mask)
                        not_mask = ~mask
                        print(ams[i][mask][:, not_mask].shape)
                        mask_idx = torch.where(mask)[0]
                        not_mask_idx = torch.where(not_mask)[0]

                        ams[i][mask_idx[:, None], not_mask_idx] = 0
                        ams[i][not_mask_idx[:, None], mask_idx] = 0

            # fig, axes = plt.subplots(1, 2)
            # random_inds = np.random.choice(np.arange(diff_10.shape[0] * diff_10.shape[1]), 1000, replace=False)
            # axes[0].hist(r0_dm.flatten().numpy()[random_inds])
            # axes[1].hist(r1_dm.flatten().numpy()[random_inds])
            # plt.show()
            # fig, axes = plt.subplots(2, 2)
            # axes[0, 0].imshow(r0_dm)
            # axes[0, 1].imshow(r1_dm)
            # vmin = min(diff_01.min(), diff_10.min())
            # vmax = max(diff_01.max(), diff_10.max())
            # axes[1, 0].imshow(diff_01, cmap='bwr', vmin=vmin, vmax=vmax)
            # axes[1, 1].imshow(diff_10, cmap='bwr', vmin=vmin, vmax=vmax)
            # plt.show()
            return dict(am_10=am_10, am_01=am_01, dm_10=dm_10, dm_01=dm_01, r0_am=r0_am, r1_am=r1_am, r0_dm=r0_dm, r1_dm=r1_dm,
                        diff_10=diff_10, diff_01=diff_01)

    @staticmethod
    def compute_rdc_sim(repr0, repr1, params):
        beta = params.get('beta', 5)
        num_samples = repr0.shape[0]
        gamma = params.get('gamma', None)
        if gamma is None:
            gamma_scale = params.get('gamma_scale', 80)
            gamma = gamma_scale / num_samples

        sim_function = params.get('sim_function', 'neighborhood')
        if sim_function == 'neighborhood':
            repr0 = torch.FloatTensor(repr0)
            repr1 = torch.FloatTensor(repr1)

            r0_dm = torch.cdist(repr0, repr0)
            r1_dm = torch.cdist(repr1, repr1)

            r0_sort = r0_dm.argsort()
            r0_dm = torch.argsort(r0_sort).float() + 1
            r1_sort = r1_dm.argsort()
            r1_dm = torch.argsort(r1_sort).float() + 1

            r1_am = torch.exp(-beta * r1_dm)
            r0_am = torch.exp(-beta * r0_dm)

            return dict(r0_am=r0_am, r1_am=r1_am, r0_dm=r0_dm, r1_dm=r1_dm)


    @staticmethod
    def cluster_graph(graph_dict, cluster_params):
        method = cluster_params.get('clustering_method', 'spectral')
        add_null_cluster = cluster_params.get('add_null_cluster', False)
        n_clusters = cluster_params.get('n_clusters', 10)
        output_dict = {}
        # clusters should be generated in both directions
        graph_keys = ['am_10', 'am_01']
        for graph_key in graph_keys:
            output_dict[graph_key] = {}
            if method == 'spectral':
                cluster = SpectralClustering(n_clusters=n_clusters + int(add_null_cluster), affinity='precomputed')
                am = graph_dict[graph_key]
                cluster.fit((am + am.T) / 2)
                RDX.post_process(cluster, graph_dict[graph_key])
                output_dict[graph_key]['cluster_labels'] = cluster.labels_

            elif method == 'affinity':
                cluster = AffinityPropagation()
                am = graph_dict[graph_key]
                cluster.fit((am + am.T) / 2)
                output_dict[graph_key]['cluster_labels'] = cluster.labels_

            elif method == 'eig_centrality':
                # eig_centrality to pick central points for "interesting" neighborhoods
                import networkx
                cluster_size = cluster_params.get("cluster_size", 9)
                am = graph_dict[graph_key]
                G = networkx.from_numpy_array(am.numpy())
                eig_centrality = networkx.eigenvector_centrality_numpy(G, weight="weight")
                arr = np.array(list(eig_centrality.values()))
                # sampling
                cluster_centers = []
                cluster_labels = np.zeros(am.shape[0], dtype=int)
                mat = am.clone()
                while len(cluster_centers) < n_clusters:
                    max_idx = np.argmax(arr)
                    cluster_centers.append(max_idx)
                    mat[max_idx, :].topk(cluster_size)
                    cl_inds = mat[max_idx, :].topk(cluster_size - 1).indices
                    mat[:, cl_inds] = -1
                    mat[:, max_idx] = -1
                    # print(max_idx, cl_inds)
                    cluster_labels[max_idx] = len(cluster_centers)
                    cluster_labels[cl_inds] = len(cluster_centers)
                    arr[max_idx] = -1
                    arr[cl_inds.numpy()] = -1
                # print(np.unique(cluster_labels, return_counts=True))
                output_dict[graph_key]['cluster_labels'] = cluster_labels

            elif method == 'pagerank':
                import networkx
                cluster_size = cluster_params.get("cluster_size", 9)
                am = graph_dict[graph_key]
                G = networkx.from_numpy_array(am.numpy())
                pagerank_out = networkx.pagerank(G, weight="weight")
                arr = np.array(list(pagerank_out.values()))
                # sampling
                cluster_centers = []
                cluster_labels = np.zeros(am.shape[0], dtype=int)
                mat = am.clone()
                while len(cluster_centers) < n_clusters:
                    max_idx = np.argmax(arr)
                    cluster_centers.append(max_idx)
                    mat[max_idx, :].topk(cluster_size)
                    cl_inds = mat[max_idx, :].topk(cluster_size - 1).indices
                    mat[:, cl_inds] = -1
                    mat[:, max_idx] = -1
                    # print(max_idx, cl_inds)
                    cluster_labels[max_idx] = len(cluster_centers)
                    cluster_labels[cl_inds] = len(cluster_centers)
                    arr[max_idx] = -1
                    arr[cl_inds.numpy()] = -1
                # print(np.unique(cluster_labels, return_counts=True))
                output_dict[graph_key]['cluster_labels'] = cluster_labels

            else:
                raise ValueError(f"Unknown clustering method {method}")

        return output_dict

    @staticmethod
    def post_process(cl, am, thresh=-1):
        labels = cl.labels_
        num_clusters = len(np.unique(labels))
        cluster_aff_mean = []
        for label_i in np.unique(labels):
            mask = labels == label_i
            cluster_aff_mean.append(am[mask][:, mask].mean())

        cluster_aff_mean = np.array(cluster_aff_mean)
        sorted_clusters = np.argsort(cluster_aff_mean)
        new_labels = np.zeros_like(labels)
        for i, cli in enumerate(sorted_clusters):
            if cluster_aff_mean[cli] < thresh:
                new_labels[labels == cli] = 0
            else:
                new_labels[labels == cli] = i
        print(new_labels)

        for label_i in np.unique(new_labels):
            mask = new_labels == label_i
            print(label_i, am[mask][:, mask].mean())
        cl.labels_ = new_labels

    @staticmethod
    def generate_visualizations(input_dict, output_dict, plot_params=None):
        output_folder = output_dict.get('method_dir', None)
        viz_output_base_folder = f'{output_folder}/visualizations'
        os.makedirs(viz_output_base_folder, exist_ok=True)

        plot_params = {} if plot_params is None else plot_params
        fig_paths = {}
        fontsize = plot_params.get('fontsize', 12)
        show, save = plot_params.get('show', False), plot_params.get('save', False)
        label_cluster_images = plot_params.get('label_cluster_images', False)
        skip_cluster_viz = plot_params.get('skip_cluster_viz', False)

        r0_red = input_dict['red0']
        r1_red = input_dict['red1']
        dataset_labels = input_dict['dataset_labels']
        cluster_dict = output_dict['cluster_dict']
        graph_dict = output_dict['graph_dict']
        representations = input_dict['representations']
        mapped_repr = False
        if input_dict.get("r0m_red") is not None:
            r0m_red = input_dict['r0m_red']
            r1m_red = input_dict['r1m_red']
            mapped_repr = True
        image_samples = input_dict['image_samples']
        image_samples_is_paths = type(image_samples[0]) == np.str_
        basic_transform = torchvision.transforms.Resize((224, 224))
        add_null_cluster = input_dict.get('add_null_cluster', False)

        r0_dm = torch.cdist(representations[0], representations[0])
        r1_dm = torch.cdist(representations[1], representations[1])

        ##### GT LABEL FIGURE START #####
        if mapped_repr:
            fig, axes = plt.subplots(3, 2, squeeze=False, figsize=(12, 12))
        else:
            fig, axes = plt.subplots(1, 2, squeeze=False)
            fig.set_size_inches(12, 6)
        cmap, get_marker, legend_plot = ph.make_large_cmap(len(np.unique(dataset_labels)))
        for labi, lab in enumerate(np.unique(dataset_labels)):
            mask = dataset_labels == lab
            c = cmap([labi] * mask.sum())
            alpha = 0.5
            marker = get_marker(labi)
            axes[0, 0].scatter(r0_red[mask, 0], r0_red[mask, 1], c=c, alpha=alpha, marker=marker, label=f'{lab}')
            axes[0, 1].scatter(r1_red[mask, 0], r1_red[mask, 1], c=c, alpha=alpha, marker=marker)
            if mapped_repr:
                axes[1, 0].scatter(r0m_red[mask, 0], r0m_red[mask, 1], c=c, alpha=alpha, marker=marker)
                axes[1, 1].scatter(r1_red[mask, 0], r1_red[mask, 1], c=c, alpha=alpha, marker=marker)
                axes[2, 0].scatter(r0_red[mask, 0], r0_red[mask, 1], c=c, alpha=alpha, marker=marker)
                axes[2, 1].scatter(r1m_red[mask, 0], r1m_red[mask, 1], c=c, alpha=alpha, marker=marker)
                axes[1, 0].set_title(f'{"Model 0 Mapped"}', fontsize=fontsize)
                axes[1, 1].set_title(f'{"Model 1 Mapped"}', fontsize=fontsize)

        axes[0, 0].set_title(f'{"Model 0"}', fontsize=fontsize)
        axes[0, 1].set_title(f'{"Model 1"}', fontsize=fontsize)
        axes[0, 0].set_xlabel('PC 1', fontsize=fontsize)
        axes[0, 0].set_ylabel('PC 2', fontsize=fontsize)
        axes[0, 1].set_xlabel('PC 1', fontsize=fontsize)
        axes[0, 1].set_ylabel('PC 2', fontsize=fontsize)
        if len(np.unique(dataset_labels)) < 35:
            axes[0, 0].legend()
        plt.tight_layout()
        ph.finish_plot(show, save, save_path=f'{viz_output_base_folder}/pca_with_gt_labels.png')

        names_dirs = [['Model 1', 'Model 0'], ['Model 0', 'Model 1']]
        dirs = ['10', '01']
        if mapped_repr:
            plot_repr = [[r0m_red, r1_red], [r0_red, r1m_red]]
        else:
            plot_repr = [[r0_red, r1_red], [r0_red, r1_red]]
        for di, d in enumerate(dirs):
            fig_paths[d] = {}
            viz_output_folder = os.path.join(viz_output_base_folder, d)
            os.makedirs(viz_output_folder, exist_ok=True)
            n0, n1 = names_dirs[di]

            cl_labels = cluster_dict[f'am_{d}']['cluster_labels']
            affinity_mat = graph_dict[f'am_{d}']
            diff_mat = graph_dict[f'diff_{d}']

            cluster_list = np.unique(cl_labels)
            num_clusters = len(cluster_list)
            cmap, get_marker, legend_plot = ph.make_large_cmap(num_clusters)
            if legend_plot is not None:
                ph.finish_plot(show, save, f'{viz_output_folder}/legend.png', fig=legend_plot)

            ##### OVERVIEW FIGURE START #####
            fig, axes = plt.subplots(2, 3)
            fig.set_size_inches(18, 12)
            r0_2d, r1_2d = plot_repr[di]
            mean_cluster_affinity = []
            for cli in np.unique(cl_labels):
                mask = cl_labels == cli
                mean_affinity = affinity_mat[mask][:, mask].mean().item()
                if cli == 0 and mean_affinity < plot_params.get('null_thresh', 0):
                    c = np.zeros((mask.sum(), 4)) + 0.5
                    alpha = 0.2
                else:
                    c = cmap(cl_labels[mask])
                    alpha = 1
                mean_cluster_affinity.append(mean_affinity)
                marker = get_marker(cli)
                axes[0, 0].scatter(r0_2d[mask, 0], r0_2d[mask, 1], c=c, alpha=alpha, marker=marker, label=f'{cli}')
                axes[1, 0].scatter(r1_2d[mask, 0], r1_2d[mask, 1], c=c, alpha=alpha, marker=marker)

            axes[0, 0].set_title(f'{names_dirs[1][0]}', fontsize=fontsize)
            axes[1, 0].set_title(f'{names_dirs[1][1]}', fontsize=fontsize)
            axes[0, 0].set_xlabel('PC 1', fontsize=fontsize)
            axes[0, 0].set_ylabel('PC 2', fontsize=fontsize)
            axes[0, 1].set_xlabel('PC 1', fontsize=fontsize)
            axes[0, 1].set_ylabel('PC 2', fontsize=fontsize)
            if len(np.unique(cl_labels)) < 35:
                axes[0, 0].legend()

            inds = []
            breaks = []
            # cl_dists = {}
            cl_affins = {}
            cl_diffs = {}
            cl_inds = {}
            for cli in np.unique(cl_labels):
                _cl_inds = np.where(cl_labels == cli)[0]
                inds.extend(_cl_inds)
                cl_inds[cli] = _cl_inds
                breaks.append(len(inds))
                cl_diffs[cli] = diff_mat[_cl_inds][:, _cl_inds]
                cl_affins[cli] = affinity_mat[_cl_inds][:, _cl_inds]
                # cl_dists[cli] = rn_dist_mat[_cl_inds][:, _cl_inds]

            inds = np.array(inds)
            axes[0, 1].imshow(r0_dm[inds][:, inds])
            axes[0, 1].set_title(f'{names_dirs[1][0]} Distance Matrix', fontsize=fontsize)
            axes[1, 1].imshow(r1_dm[inds][:, inds])
            axes[1, 1].set_title(f'{names_dirs[1][1]} Distance Matrix', fontsize=fontsize)

            diff_mat_min, diff_mat_max = diff_mat.min(), diff_mat.max()
            im = axes[0, 2].imshow(diff_mat[inds][:, inds], cmap='bwr', vmin=diff_mat_min, vmax=diff_mat_max)
            axes[0, 2].set_title(f'{n0} - {n1} Difference Matrix')
            fig.colorbar(im, ax=axes[0, 2])

            norm = None
            if affinity_mat.abs().max() / affinity_mat.abs().min() > 100:
                norm = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=affinity_mat.min(), vmax=affinity_mat.max())
            im = axes[1, 2].imshow(affinity_mat[inds][:, inds], cmap='viridis', norm=norm)
            axes[1, 2].set_title('$M_A$')
            fig.colorbar(im, ax=axes[1, 2])

            axes_w_clus_bound = [axes[0, 1], axes[1, 1], axes[0, 2], axes[1, 2]]
            for bi, b in enumerate(breaks):
                b = b - 0.5
                prev = breaks[bi - 1] if bi > 0 else 0
                for ax in axes_w_clus_bound:
                    ax.plot([prev, b], [prev, prev], color='m')
                    ax.plot([prev, b], [b, b], color='m')
                    ax.plot([prev, prev], [prev, b], color='m')
                    ax.plot([b, b], [prev, b], color='m')
                # axes[1, 1].plot([prev, b], [prev, prev], color='m')
                # axes[1, 1].plot([prev, b], [b, b], color='m')
                # axes[1, 1].plot([prev, prev], [prev, b], color='m')
                # axes[1, 1].plot([b, b], [prev, b], color='m')
                plt.tight_layout()

            ph.finish_plot(show, save, save_path=f'{viz_output_folder}/overview.png')
            if save:
                fig_paths[d]['overview'] = f'{viz_output_folder}/overview.png'

            ##### OVERVIEW FIGURE END #####

            ##### CLUSTER FIGURE START #####
            # subsample from cluster to generate grids
            grid_size = plot_params.get('grid_size', '4x4')
            axes_pad = plot_params.get('axes_pad', 0.3)
            cluster_sample_strategy = plot_params.get('cluster_sample_strategy', 'centroid')
            indices = np.arange(0, r0_red.shape[0])
            num_samples = plot_params.get('num_samples', 16)
            explanations_folder = f'{viz_output_folder}/explanations'
            os.makedirs(explanations_folder, exist_ok=True)
            fig_paths[d]['clusters'] = []

            selected_indices = []

            for i in cluster_list:
                mask = cl_labels == i
                num_items = mask.sum()
                if cluster_sample_strategy == 'random':
                    inds = np.where(mask)[0]
                    sel_inds = np.random.choice(inds, min(num_items, num_samples), replace=False)
                elif cluster_sample_strategy == 'max_affinity':
                    cl_affin = affinity_mat[mask][:, mask].mean(1)
                    cl_inds = cl_affin.argsort(descending=True)
                    sel_inds = indices[mask][cl_inds][:num_samples]
                elif cluster_sample_strategy == 'max_affinity_spectral_neighborhood':
                    cl_affin = affinity_mat[mask][:, mask].mean(1)
                    cl_anch = cl_affin.argsort(descending=True)[0]
                    curr_repr = representations[1] if d == '10' else representations[0]
                    anc_point = curr_repr[mask][cl_anch]
                    sel_inds = np.where(mask)[0][
                        torch.cdist(anc_point[None, :], curr_repr[mask])[0].argsort()[:num_samples].numpy()]
                elif cluster_sample_strategy == 'max_affinity_neighborhood':
                    # TODO delete this after converting code
                    cl_affin = affinity_mat[mask][:, mask].mean(1)
                    cl_anch = cl_affin.argsort(descending=True)[0]
                    curr_repr = representations[1] if d == '10' else representations[0]
                    anc_point = curr_repr[mask][cl_anch]
                    sel_inds = np.where(mask)[0][
                        torch.cdist(anc_point[None, :], curr_repr[mask])[0].argsort()[:num_samples].numpy()]
                elif cluster_sample_strategy == 'maximize_total_neighborhood_affinity':
                    cl_inds = np.where(mask)[0]
                    cl_affin = affinity_mat[mask][:, mask]
                    curr_repr = representations[1] if d == '10' else representations[0]
                    if num_items <= 1:
                        sel_inds = cl_inds
                    else:
                        cl_dists = torch.cdist(curr_repr[cl_inds], curr_repr[cl_inds])
                        neighborhoods = torch.topk(cl_dists, min(num_items, num_samples), dim=1, largest=False)[1]
                        nsums = []
                        for ni in range(len(neighborhoods)):
                            nsums.append(cl_affin[neighborhoods[ni]][:, neighborhoods[ni]].sum())
                        nsums = torch.stack(nsums)
                        nb_ind = torch.argmax(nsums)
                        sel_inds = cl_inds[neighborhoods[nb_ind]]
                    # cl_affin_argsort = cl_affin.argsort(dim=1, descending=True)
                    # cl_affin_neighborhood_sum = torch.gather(cl_affin, dim=1, index=cl_affin_argsort[:, :num_samples]).sum(1)
                    # cl_anch = cl_affin_neighborhood_sum.argmax()
                    # sel_inds = cl_inds[cl_affin_argsort[cl_anch, :(num_samples - 1)]]
                    # sel_inds = np.concatenate(([cl_inds[cl_anch]], sel_inds))

                elif cluster_sample_strategy == 'spectral_centrality':
                    cl_inds = np.where(mask)[0]
                    cl_affin = affinity_mat[mask][:, mask]
                    lap = sp.sparse.csgraph.laplacian(cl_affin.numpy())
                    # vals, vecs = torch.linalg.eig(cl_affin)
                    vals, vecs = sparse_linalg.eigs(lap, k=1, which='LR')
                    sel_inds = cl_inds[vecs[:, 0].argsort()[::-1][:num_samples]]
                elif cluster_sample_strategy == 'max_affinity_euclidean_neighborhood':
                    cl_affin = affinity_mat[mask][:, mask].mean(1)
                    cl_anch = cl_affin.argsort(descending=True)[0]
                    curr_repr = representations[1] if d == '10' else representations[0]
                    anc_point = curr_repr[mask][cl_anch]
                    sel_inds = torch.cdist(anc_point[None, :], curr_repr).argsort()[:num_samples].numpy()
                elif cluster_sample_strategy == 'spectral_cluster_centroid':
                    curr_repr = representations[1] if d == '10' else representations[0]
                    model_centroid = curr_repr[mask].mean(0)
                    sel_inds = np.where(mask)[0][
                        torch.cdist(model_centroid[None, :], curr_repr[mask])[0].argsort()[:num_samples].numpy()]
                elif cluster_sample_strategy == 'centroid':
                    curr_repr = representations[1] if d == '10' else representations[0]
                    model_centroid = curr_repr[mask].mean(0)
                    sel_inds = torch.cdist(model_centroid[None, :], curr_repr)[0].argsort()[:num_samples].numpy()
                else:
                    raise ValueError(f"Unknown cluster sampling strategy {cluster_sample_strategy}")
                selected_indices.append(sel_inds)
                if image_samples_is_paths:
                    sel_paths = image_samples[sel_inds]
                    sel_images = [basic_transform(Image.open(p)) for p in sel_paths]
                else:
                    sel_images = image_samples[sel_inds].permute(0, 2, 3, 1).cpu().numpy()

                if not skip_cluster_viz:
                    fig, grid = ph.make_image_grid(sel_images, mode=grid_size, axes_pad=axes_pad)
                    if label_cluster_images:
                        for axi, ax in enumerate(grid):
                            ax.set_title(f'{dataset_labels[sel_inds[axi]]}', fontsize=fontsize)

                    # plt.suptitle(f'Cluster {i}', fontsize=fontsize)
                    plt.tight_layout()
                    # plt.show()
                    fig_path = f'{explanations_folder}/{i}.png'
                    ph.finish_plot(show, save, save_path=fig_path, fig=fig)
                    fig_paths[d]['clusters'].append(fig_path)

            fig_paths[d]['selected_indices'] = selected_indices
            fig_paths[d]['mean_cluster_affinity'] = mean_cluster_affinity
            # summarize cluster figures in a single figure
            pad = plot_params.get('cl_summ_pad', 10)
            summ_size = 600
            bg_color = plot_params.get('cl_summ_bg_color', [0, 0, 0])
            bg_color = tuple(bg_color) if type(bg_color) == list else bg_color

            if not skip_cluster_viz:
                summ_figs = fig_paths[d]['clusters']
                if plot_params.get('skip_low_affinity_for_summary', False):
                    summ_figs = [f for i, f in enumerate(summ_figs) if mean_cluster_affinity[i] > plot_params.get('null_thresh', 0)]
                if add_null_cluster:
                    summ_figs = summ_figs[1:]

                pad_mult = 1
                if len(summ_figs) != 0 and save:
                    cl_summ_im = Image.new('RGB', (len(summ_figs) * (summ_size + pad_mult*pad) - (pad_mult-2) * pad,
                                                   summ_size + 2 * pad),
                                           color=bg_color)
                    start_pos_x = pad
                    for i, fig_path in enumerate(summ_figs):
                        im = Image.open(fig_path)
                        im = im.resize((summ_size, summ_size))
                        cl_summ_im.paste(im, (start_pos_x, pad))
                        start_pos_x += summ_size + pad_mult * pad
                    if show:
                        fig = plt.figure()
                        fig.set_size_inches(20, 3)
                        plt.axis('off')
                        plt.imshow(cl_summ_im)
                        plt.show()
                    if save:
                        cl_summ_im.save(f'{viz_output_base_folder}/cluster_summ_{d}.png')
                        fig_paths[d]['cluster_summ'] = f'{viz_output_base_folder}/cluster_summ_{d}.png'

            ##### CLUSTER FIGURE END #####

            ##### SELECTED INDICES FIGURE START #####
            fig, axes = plt.subplots(1, 2, squeeze=False)
            fig.set_size_inches(12, 6)

            axes[0, 0].scatter(r0_2d[:, 0], r0_2d[:, 1], c='gray', alpha=0.2)
            axes[0, 1].scatter(r1_2d[:, 0], r1_2d[:, 1], c='gray', alpha=0.2)

            for cli in range(len(selected_indices)):
                curr_si = selected_indices[cli]
                c = cmap(cl_labels[curr_si])
                alpha = 1
                marker = get_marker(cli)
                axes[0, 0].scatter(r0_2d[curr_si, 0], r0_2d[curr_si, 1], c=c, alpha=alpha, marker=marker, label=f'{cli}')
                axes[0, 1].scatter(r1_2d[curr_si, 0], r1_2d[curr_si, 1], c=c, alpha=alpha, marker=marker)

            axes[0, 0].set_title(f'{names_dirs[1][0]}', fontsize=fontsize)
            axes[0, 1].set_title(f'{names_dirs[1][1]}', fontsize=fontsize)
            axes[0, 0].set_xlabel('PC 1', fontsize=fontsize)
            axes[0, 0].set_ylabel('PC 2', fontsize=fontsize)
            axes[0, 1].set_xlabel('PC 1', fontsize=fontsize)
            axes[0, 1].set_ylabel('PC 2', fontsize=fontsize)
            if len(np.unique(cl_labels)) < 35:
                axes[0, 0].legend()
            plt.tight_layout()
            fig_path = f'{viz_output_folder}/sel_inds_viz.png'
            ph.finish_plot(show, save, save_path=fig_path, fig=fig)
            fig_paths[d]['sel_inds_viz'] = fig_path

            ##### SELECTED INDICES FIGURE END #####

        return fig_paths