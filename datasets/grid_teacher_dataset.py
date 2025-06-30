import os
import torch
from PIL import Image
import json
import numpy as np
from src.utils import model_loader
from tqdm import tqdm
from src.rdc import construct_graph
from src.cka import CudaCKA
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from src.utils.plotting_helper import make_image_grid
from datasets.butterflies import ButterfliesDataset
from datasets.nabirds import NABirds


def weighted_choice(items, dist, size, device=None, force_sum_to_one=True):
    """Do inverse transform sampling to sample from a grid with given probabilities
    https://en.wikipedia.org/wiki/Inverse_transform_sampling

    Args:
        items (tensor): Random are samples pulled from this 1D tensor
        dist (tensor): Probability of each item in the grid.  Should sum to 1
        size (tuple): shape of sampled output
        device (str, optional): pytorch device to use

    Returns:
        sampled (tensor): sampled values from `items` with shape `size`
    """
    items = torch.as_tensor(items, device=device)
    dist = torch.as_tensor(dist, device=device)
    if force_sum_to_one:
        # handles any numerical issues
        dist = dist / dist.sum()

    dist_cum = torch.cumsum(dist, 0)
    rand_ind = torch.searchsorted(dist_cum, torch.rand(size=size, device=device), side='right')
    return items[rand_ind]




class GridTeacherDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, root, teacher_model, student_model, grid_sampling_config, split=None, transform=None,
                 dataset_loading_params=None):
        self.dataset_name = dataset_name
        self.labels = None
        self.paths = None
        self.root = root
        self.transform = transform
        dataset_loading_params = {} if dataset_loading_params is None else dataset_loading_params
        dataset_loading_params['root'] = root
        self.core_dataset = self.load_dataset(dataset_name, dataset_loading_params)

        self.labels = np.array(self.labels)
        self.paths = np.array(self.paths)

        self.student_model = student_model.cuda()
        self.teacher_model = teacher_model.cuda()
        if self.student_model is not None:
            self.teacher_embeddings, self.student_embeddings = self.get_embeddings(
                [self.teacher_model, self.student_model])
        else:
            self.teacher_embeddings, = self.get_embeddings([self.teacher_model])
            self.student_embeddings = None

        self.grids = None
        self.cluster_sims = None
        self.loss_flags = None
        self.sampling_figs = {}

        self.grid_sampling_config = grid_sampling_config
        key = f'{split}_sampling_strategies' if split is not None else 'sampling_strategies'
        self.sample_grids(grid_sampling_config[key])

    def load_dataset(self, dataset_name, params):
        if dataset_name == 'butterflies':
            dataset = ButterfliesDataset(**params)
            self.labels = dataset.labels
            self.paths = dataset.paths
            # append root to paths
            self.paths = [os.path.join(self.root, path) for path in self.paths]
        elif dataset_name == 'nabirds':
            dataset = NABirds(**params)
            self.paths, self.labels = zip(*dataset.samples)

        return dataset

    @torch.no_grad()
    def get_embeddings(self, models, batch_size=64):
        embeddings = [[] for _ in range(len(models))]
        batch = []
        counter = 0
        for path_ind, path in enumerate(tqdm(self.paths)):
            image = self._load_and_transform_image(path_ind)['input'].cuda().unsqueeze(0)
            if counter < batch_size - 1:
                batch.append(image)
                counter += 1
            else:
                counter = 0
                batch.append(image)
                for mi, model in enumerate(models):
                    embedding = model(torch.cat(batch, dim=0))
                    embeddings[mi].append(embedding.cpu())
                batch = []

        for mi, model in enumerate(models):
            embedding = model(torch.cat(batch, dim=0))
            embeddings[mi].append(embedding.cpu())
            embeddings[mi] = torch.cat(embeddings[mi], dim=0)

        return embeddings

    def sample_grids(self, grid_sampling_params):

        grid_list = []
        clsim_list = []
        cluster_fits = []
        loss_flags_list = []
        for i, grid_params in enumerate(grid_sampling_params):
            method = grid_params['method']
            print(f'Sampling grid {i + 1}/{len(grid_sampling_params)} with method {method}')
            out = self._sample_grids(grid_params)
            grids = out['grids']
            grid_list.append(grids)

            cluster_sim = out.get('cluster_sim', None)
            if cluster_sim is None:
                cluster_sim = torch.ones((grids.shape[0], grids.shape[1]), dtype=torch.bool)
            clsim_list.append(cluster_sim)

            cluster_fit = out.get('cluster_fit', None)
            if cluster_fit is not None:
                cluster_fits.append(cluster_fit)
            grid_inds = grids
            num_grid_groups = grid_params.get('num_grid_groups', 20)
            num_grids = grid_params.get('num_grids_per_group', 10)
            loss_flags = out.get('loss_flags', None)
            if loss_flags is not None:
                loss_flags_list.append(loss_flags)
            grid_size = grid_params.get('grid_size', 16)
            for k in range(num_grid_groups):
                sets = [set(gi.tolist()) for gi in grid_inds[k]]
                for i in range(num_grids):
                    for j in range(num_grids):
                        if i == j:
                            continue
                        if sets[i].intersection(sets[j]):
                            print(i, j)

        teacher_dist = torch.cdist(self.teacher_embeddings, self.teacher_embeddings)
        teacher_sort_inds = teacher_dist.argsort()
        labels = torch.arange(num_grids).unsqueeze(1).repeat(1, grid_size)
        label_mask = torch.stack([labels.flatten() == labels.flatten()[i] for i in range(num_grids * grid_size)], dim=0)
        loss_flags_list = []
        for gi in range(grids.shape[0]):
            dmat = teacher_dist[grids[gi].flatten()][:, grids[gi].flatten()]
            max_pos_dist = dmat[label_mask].reshape(-1, grid_size).max(-1)[0]
            below_max_pos_dist_mask = dmat <= max_pos_dist.unsqueeze(1)
            loss_flags = - 1 * torch.ones_like(dmat)
            loss_flags[below_max_pos_dist_mask] = 0
            loss_flags[label_mask] = 1
            loss_flags_list.append(loss_flags)
            # print((loss_flags[np.arange(0, 160, 16)] == 0).sum())

        self.loss_flags = torch.stack(loss_flags_list, dim=0)
        # plt.figure()
        # plt.imshow(self.loss_flags[0], cmap='bwr')
        # plt.show()
        # torch.set_printoptions(sci_mode=False)

        self.grids = torch.cat(grid_list, dim=0)
        # self.loss_flags = torch.cat(loss_flags_list, dim=0) if len(loss_flags_list) > 0 else None
        self.cluster_sims = torch.cat(clsim_list, dim=0)
        print(f'Sampled {len(self.grids)} grids')

        viz_cluster_fit = len(cluster_fits) > 0 and grid_sampling_params[0]['method'] == 'rdc'
        ngg, ng, gs = self.grids.shape
        fig, axes = plt.subplots(1, 2 + int(viz_cluster_fit), figsize=(10 + 5 * int(viz_cluster_fit), 5))
        red = PCA(n_components=2)
        emb2d = red.fit_transform(self.teacher_embeddings)
        emb2d_stud = red.fit_transform(self.student_embeddings)
        for lab_i in np.unique(self.labels):
            mask = self.labels == lab_i
            axes[0].scatter(emb2d[mask, 0], emb2d[mask, 1], cmap='tab10')
            axes[1].scatter(emb2d_stud[mask, 0], emb2d_stud[mask, 1], cmap='tab10')
        if viz_cluster_fit:
            clabels = np.unique(cluster_fits[0].labels_)[clsim_list[0].argsort()]
            for clab_i in clabels:
                mask = cluster_fits[0].labels_ == clab_i
                axes[2].scatter(emb2d[mask, 0], emb2d[mask, 1], cmap='tab10')
            axes[2].set_title('Clustered Teacher Embeddings')

        anchor_inds = self.grids.reshape(-1, gs)[:, 0]
        for anchor_ind in anchor_inds:
            axes[0].scatter(emb2d[anchor_ind, 0], emb2d[anchor_ind, 1], s=100, marker='x', c='k', alpha=0.6)
            axes[1].scatter(emb2d_stud[anchor_ind, 0], emb2d_stud[anchor_ind, 1], s=100, marker='x', c='k', alpha=0.6)
            if viz_cluster_fit:
                axes[2].scatter(emb2d[anchor_ind, 0], emb2d[anchor_ind, 1], s=100, marker='x', c='k', alpha=0.6)
        axes[0].set_title('Teacher Embeddings')
        axes[1].set_title('Student Embeddings')
        plt.tight_layout()
        self.sampling_figs['anchors'] = fig

        # if viz_cluster_fit:
            # fig, axes = plt.subplots(1, 1)
            # _, counts = np.unique(cluster_fits[0].labels_, return_counts=True)
            # counts = counts[clsim_list[0].argsort()]
            # axes.bar(np.arange(len(counts)), counts)
            # axes.set_title('Cluster Sizes')
            # self.sampling_figs['cluster_sizes'] = fig

        if self.grid_sampling_config.get("sample_grid_viz_path", None) is not None:
            outpath = self.grid_sampling_config['sample_grid_viz_path']
            if self.split is None:
                outpath = os.path.join(outpath, 'all')
            else:
                outpath = os.path.join(outpath, self.split)
            os.makedirs(outpath, exist_ok=True)
            for ci in range(ng):
                im_inds = self.grids[0, ci]
                images = [Image.open(os.path.join(self.paths[im_ind])).convert('RGB') for im_ind in im_inds]
                fig, axes = make_image_grid(images, mode="4x4")
                for axi, ax in enumerate(axes):
                    label = self.labels[im_inds][axi]
                    ax.set_title(f'{im_inds[axi]} | {label}')
                plt.suptitle(f"Cluster {ci}")
                plt.savefig(os.path.join(outpath, f"cluster_{ci}.png"))

    def _sample_grids(self, grid_sampling_params):
        method = grid_sampling_params['method']
        num_grid_groups = grid_sampling_params.get('num_grid_groups', 20)
        num_grids = grid_sampling_params.get('num_grids_per_group', 10)
        grid_size = grid_sampling_params.get('grid_size', 16)
        indices = np.arange(len(self.teacher_embeddings))
        out = {}
        if method == 'random':
            teacher_dist = torch.cdist(self.teacher_embeddings, self.teacher_embeddings)
            teacher_nn = teacher_dist.topk(k=grid_size, dim=1, largest=False)[1]

            max_loops = 100
            tmp = []
            for i in range(num_grid_groups):
                grid_group_hist = set()
                gc = 0
                grids = []
                num_iters = 0
                while gc < num_grids:
                    init_ind = np.random.choice(indices, size=1)
                    grid = teacher_nn[init_ind].squeeze()
                    if grid_group_hist.isdisjoint(set(grid.tolist())):
                        grid_group_hist.update(set(grid.tolist()))
                        grids.append(grid)
                        gc += 1
                    num_iters += 1
                    if num_iters > max_loops:
                        print('max loops reached, change grid size, model or number of images')
                        exit(0)

                grids = torch.stack(grids, dim=0)
                tmp.append(grids)
            grids = torch.stack(tmp, dim=0)
            # replace = True if num_grid_groups * num_grids > len(indices) else False
            # init_inds = np.random.choice(indices, size=num_grid_groups * num_grids, replace=replace)
            # grids = teacher_nn[init_inds].reshape(num_grid_groups, num_grids, grid_size)
            # grid_valid = []
            out['grids'] = grids
        elif method == 'kmeans':
            sample_strategy = grid_sampling_params.get('sampling_strategy', 'random')

            if sample_strategy == 'random':
                cl = KMeans(n_clusters=num_grids, random_state=0)
                cl.fit(self.teacher_embeddings)
                grid_list = []
                for i in range(num_grid_groups):
                    _gg = []
                    for j in range(num_grids):
                        replace = True if grid_size > len(indices[cl.labels_ == j]) else False
                        grid = np.random.choice(indices[cl.labels_ == j], size=grid_size, replace=replace)
                        _gg.append(grid)
                    grid_list.append(_gg)
                grids = np.array(grid_list)
                grids = torch.tensor(grids)
                out['grids'] = grids
                out['cluster_fit'] = cl

            elif sample_strategy == 'neighborhood':
                cl = KMeans(n_clusters=num_grids, random_state=0)
                cl.fit(self.teacher_embeddings)
                teacher_dist = torch.cdist(self.teacher_embeddings, self.teacher_embeddings)
                grid_list = []
                for j in range(num_grids):
                    cl_inds = indices[cl.labels_ == j]
                    _int_cl_inds = torch.arange(len(cl_inds))
                    replace = True if num_grid_groups > len(cl_inds) else False
                    init_inds = np.random.choice(_int_cl_inds, size=num_grid_groups, replace=replace)
                    k = min(grid_size, len(cl_inds))
                    cl_nn = teacher_dist[cl_inds][:, cl_inds].topk(k=k, dim=1, largest=False)[1]
                    cl_grids = cl_nn[init_inds]
                    cluster_grids = cl_inds[cl_grids]
                    grid_list.append(cluster_grids)
                grids = np.array(grid_list)
                grids = torch.tensor(grids).permute(1, 0, 2)
                out['grids'] = grids
                out['cluster_fit'] = cl

            elif sample_strategy == 'neighborhood_v2':
                cl = KMeans(n_clusters=num_grids, random_state=0)
                cl.fit(self.teacher_embeddings)
                teacher_dist = torch.cdist(self.teacher_embeddings, self.teacher_embeddings)
                grid_list = []
                for j in range(num_grids):
                    cl_inds = indices[cl.labels_ == j]
                    _int_cl_inds = torch.arange(len(cl_inds))
                    replace = True if num_grid_groups > len(cl_inds) else False
                    init_inds = np.random.choice(_int_cl_inds, size=num_grid_groups, replace=replace)
                    cl_nn = teacher_dist[cl_inds].topk(k=grid_size, dim=1, largest=False)[1]
                    cluster_grids = cl_nn[init_inds]
                    grid_list.append(cluster_grids)
                grids = np.array(grid_list)
                grids = torch.tensor(grids).permute(1, 0, 2)
                out['grids'] = grids
                out['cluster_fit'] = cl

        elif method == 'rdc':
            beta = grid_sampling_params.get('beta', 5)
            rdc_aff_mean_thresh = grid_sampling_params.get('rdc_aff_mean_thresh', 1.5)
            graph_dict = construct_graph(self.student_embeddings, self.teacher_embeddings, beta=beta, gamma_scale=40)
            am_ts = graph_dict['am_ts']
            am_ts = (am_ts + am_ts.T) / 2
            diff_ts = graph_dict['diff_ts']
            tr_sort = graph_dict['tr_sort']
            sample_strategy = grid_sampling_params.get('sampling_strategy', 'spectral_random')

            if sample_strategy == 'spectral_random':
                cl = SpectralClustering(n_clusters=num_grids, affinity='precomputed').fit(am_ts)
                cluster_sim = []
                for label in np.unique(cl.labels_):
                    mask = cl.labels_ == label
                    aff_mean = am_ts[mask][:, mask].mean()
                    cluster_sim.append(aff_mean)
                cluster_sim = torch.tensor(cluster_sim)
                cluster_sim = cluster_sim.unsqueeze(0).repeat(num_grid_groups, 1).type(torch.bool)

                grid_list = []
                for i in range(num_grid_groups):
                    _gg = []
                    for j in range(num_grids):
                        grid = np.random.choice(indices[cl.labels_ == j], size=grid_size, replace=True)
                        _gg.append(grid)
                    grid_list.append(_gg)
                grids = np.array(grid_list)
                grids = torch.tensor(grids)
                out['grids'] = grids
                out['cluster_sim'] = cluster_sim
                out['cluster_fit'] = cl

            if sample_strategy == 'spectral_neighborhood':
                cl = SpectralClustering(n_clusters=num_grids, affinity='precomputed').fit(am_ts)
                cluster_sim = []
                for label in np.unique(cl.labels_):
                    mask = cl.labels_ == label
                    aff_mean = am_ts[mask][:, mask].mean()
                    cluster_sim.append(aff_mean)
                cluster_sim = torch.tensor(cluster_sim)
                # cluster_sim = cluster_sim.unsqueeze(0).repeat(num_grid_groups, 1).type(torch.bool)

                teacher_dist = torch.cdist(self.teacher_embeddings, self.teacher_embeddings)
                grid_list = []
                for j in range(num_grids):
                    cl_inds = indices[cl.labels_ == j]
                    tmp = am_ts[cl_inds][:, cl_inds]
                    p = torch.softmax(tmp.mean(1), dim=-1).cpu().numpy() if grid_sampling_params.get('weighted_anchor', False) else None
                    _int_cl_inds = torch.arange(len(cl_inds))
                    replace = True if num_grid_groups > len(cl_inds) else False
                    init_inds = np.random.choice(_int_cl_inds, p=p, size=num_grid_groups, replace=replace)
                    k = min(grid_size, len(cl_inds))
                    cl_nn = teacher_dist[cl_inds][:, cl_inds].topk(k=k, dim=1, largest=False)[1]
                    if cl_nn.shape[1] < grid_size:
                        # repeat the last element to fill the grid
                        size_diff = grid_size - cl_nn.shape[1]
                        cl_nn = torch.cat([cl_nn, cl_nn[:, -1].unsqueeze(1).repeat(1, size_diff)], dim=1)

                    cl_grids = cl_nn[init_inds]
                    cluster_grids = cl_inds[cl_grids]
                    grid_list.append(cluster_grids)
                grids = np.array(grid_list)
                grids = torch.tensor(grids).permute(1, 0, 2)
                out['grids'] = grids
                out['cluster_sim'] = cluster_sim
                out['cluster_fit'] = cl

            if sample_strategy == 'spectral_neighborhood_v2':
                cl = SpectralClustering(n_clusters=num_grids, affinity='precomputed').fit(am_ts)
                cluster_sim = []
                for label in np.unique(cl.labels_):
                    mask = cl.labels_ == label
                    aff_mean = am_ts[mask][:, mask].mean()
                    cluster_sim.append(aff_mean)
                cluster_sim = torch.tensor(cluster_sim)
                # cluster_sim = cluster_sim.unsqueeze(0).repeat(num_grid_groups, 1).type(torch.bool)

                teacher_dist = torch.cdist(self.teacher_embeddings, self.teacher_embeddings)
                grid_list = []
                for j in range(num_grids):
                    cl_inds = indices[cl.labels_ == j]
                    tmp = am_ts[cl_inds][:, cl_inds]
                    p = torch.softmax(tmp.mean(1), dim=-1).cpu().numpy() if grid_sampling_params.get('weighted_anchor', False) else None
                    _int_cl_inds = torch.arange(len(cl_inds))
                    replace = True if num_grid_groups > len(cl_inds) else False
                    # TODO option: add nn (not already in cluster) to centroid of cluster if cluster < grid_size
                    init_inds = np.random.choice(_int_cl_inds, p=p, size=num_grid_groups, replace=replace)
                    cl_nn = teacher_dist[cl_inds].topk(k=grid_size, dim=1, largest=False)[1]
                    # if cl_nn.shape[1] < grid_size:
                    #     # repeat the last element to fill the grid
                    #     size_diff = grid_size - cl_nn.shape[1]
                    #     cl_nn = torch.cat([cl_nn, cl_nn[:, -1].unsqueeze(1).repeat(1, size_diff)], dim=1)

                    cluster_grids = cl_nn[init_inds]
                    grid_list.append(cluster_grids)
                grids = np.array(grid_list)
                grids = torch.tensor(grids).permute(1, 0, 2)

                # plt.figure()
                # plt.imshow(loss_flags, cmap='bwr')
                # plt.show()

                out['grids'] = grids
                out['cluster_sim'] = cluster_sim
                out['cluster_fit'] = cl
                # out['loss_flags'] = loss_flags

            if sample_strategy == 'spectral_neighborhood_v3':
                cl = SpectralClustering(n_clusters=num_grids, affinity='precomputed').fit(am_ts)
                cluster_sim = []
                for label in np.unique(cl.labels_):
                    mask = cl.labels_ == label
                    aff_mean = am_ts[mask][:, mask].mean()
                    cluster_sim.append(aff_mean)
                cluster_sim = torch.tensor(cluster_sim)
                # cluster_sim = cluster_sim.unsqueeze(0).repeat(num_grid_groups, 1).type(torch.bool)

                teacher_dist = torch.cdist(self.teacher_embeddings, self.teacher_embeddings)
                grid_group_list = []

                for i in range(num_grid_groups):
                    used_images = set()
                    grid_list = []
                    for j in cluster_sim.argsort(descending=True).tolist():
                        cl_inds = indices[cl.labels_ == j]
                        not_cl_inds = indices[cl.labels_ != j]
                        # tmp = am_ts[cl_inds][:, cl_inds]
                        # p = torch.softmax(tmp.mean(1), dim=-1).cpu().numpy() if grid_sampling_params.get('weighted_anchor', False) else None
                        # _int_cl_inds = torch.arange(len(cl_inds))
                        # replace = True if num_grid_groups > len(cl_inds) else False
                        # TODO option: add nn (not already in cluster) to centroid of cluster if cluster < grid_size
                        not_cl_dist = teacher_dist[cl_inds][:, not_cl_inds].mean(0)
                        not_cl_inds = not_cl_inds[not_cl_dist.argsort()]

                        np.random.shuffle(cl_inds)
                        cl_nn = teacher_dist.topk(k=grid_size, dim=1, largest=False)[1]

                        for ii, init_ind in enumerate(np.concatenate([cl_inds, not_cl_inds])):
                            if ii >= len(cl_inds):
                                print('no valid anchor in cluster')
                            cl_nn_ii = cl_nn[init_ind]
                            if len(used_images.intersection(set(cl_nn_ii.tolist()))) == 0:
                                used_images.update(set(cl_nn_ii.tolist()))
                                grid_list.append(cl_nn_ii)
                                break
                    grid_group_list.append(grid_list)
                        # if cl_nn.shape[1] < grid_size:
                        #     # repeat the last element to fill the grid
                        #     size_diff = grid_size - cl_nn.shape[1]
                        #     cl_nn = torch.cat([cl_nn, cl_nn[:, -1].unsqueeze(1).repeat(1, size_diff)], dim=1)

                grids = np.array(grid_group_list)
                grids = torch.tensor(grids)

                # plt.figure()
                # plt.imshow(loss_flags, cmap='bwr')
                # plt.show()

                out['grids'] = grids
                out['cluster_sim'] = cluster_sim
                out['cluster_fit'] = cl
                # out['loss_flags'] = loss_flags

            if sample_strategy == 'weighted_neighborhood':
                teacher_dist = torch.cdist(self.teacher_embeddings, self.teacher_embeddings)
                teacher_nn = teacher_dist.topk(k=grid_size, dim=1,largest=False)[1]
                probs = torch.softmax(-beta * diff_ts.mean(1), dim=-1).numpy()

                tmp = []
                for i in range(num_grid_groups):
                    grid_group_hist = set()
                    gc = 0
                    grids = []
                    while gc < num_grids:
                        init_ind = np.random.choice(indices, p=probs, size=1)
                        grid = teacher_nn[init_ind].squeeze()
                        if grid_group_hist.isdisjoint(set(grid.tolist())):
                            grid_group_hist.update(set(grid.tolist()))
                            grids.append(grid)
                            gc += 1
                    grids = torch.stack(grids, dim=0)
                    tmp.append(grids)
                grids = torch.stack(tmp, dim=0)
                out['grids'] = grids

        if 'grids' not in out:
            raise NotImplementedError(f"Sampling method configuration does not exist: {grid_sampling_params}")

        # NGG, N, G = grids.shape
        # all_dists = torch.cdist(self.teacher_embeddings, self.teacher_embeddings)
        # labels = torch.arange(N).unsqueeze(1).repeat(1, G).to(all_dists.device)
        # margin = 1
        # label_mask = torch.stack([labels.flatten() == labels.flatten()[i] for i in range(N * G)], dim=0)
        # for ngi in range(NGG):
        #     dists = all_dists[grids[ngi].flatten()][:, grids[ngi].flatten()]
        #     num_zero_losses = 0
        #     for i in range(N * G):
        #         pos_dists = dists[i, label_mask[i]]
        #         pos_dists = torch.cat([pos_dists[:i], pos_dists[i + 1:]])
        #         neg_dists = dists[i, ~label_mask[i]]
        #         dist_diffs = pos_dists.unsqueeze(1) - neg_dists.unsqueeze(0)
        #         losses = torch.nn.functional.relu(dist_diffs)
        #         num_zero_losses += (losses == 0).sum().item()
        #     print(f'Num zero losses: {num_zero_losses}, {losses.mean()}')

        return out

    def __len__(self):
        return self.grids.shape[0]

    def _load_and_transform_image(self, idx):
        path = self.paths[idx]
        image = Image.open(os.path.join(path)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return dict(input=image, target=self.labels[idx])

    def __getitem__(self, item):
        grids = self.grids[item]
        labels = self.labels[grids.flatten()].reshape(grids.shape)
        images = []
        for grid in grids:
            grid_images = []
            for image_i in grid:
                image = self._load_and_transform_image(image_i)['input']
                grid_images.append(image)
            images.append(torch.stack(grid_images, dim=0))
        images = torch.stack(images, dim=0)

        if self.loss_flags is not None:
            return dict(grid_inds=grids, input=images, cluster_sims=self.cluster_sims[item], labels=labels, loss_flags=self.loss_flags[item])
        else:
            return dict(grid_inds=grids, input=images, cluster_sims=self.cluster_sims[item], labels=labels)



if __name__ == '__main__':
    sm_ckpt_path = "/home/nkondapa/PycharmProjects/ConceptDiff/checkpoints/butterflies_student1/last.ckpt"
    tm_ckpt_path = "/home/nkondapa/PycharmProjects/ConceptDiff/checkpoints/butterflies_expert/last.ckpt"
    config_path = "../grid_sampling_configs/test.json"
    with open(config_path, 'r') as f:
        grid_sampling_config = json.load(f)

    model_name = 'resnet18.a2_in1k'
    transform = model_loader.load_model(model_name)['test_transform']
    teacher_model = model_loader.load_model(model_name, tm_ckpt_path, device='cuda', eval=True)['model']
    student_model = model_loader.load_model(model_name, sm_ckpt_path, device='cuda', eval=True)['model']
    teacher_model.fc = torch.nn.Identity()
    student_model.fc = torch.nn.Identity()
    for split in ['train', 'test', None]:
        # dataset = GridTeacherDataset('butterflies', '/home/nkondapa/Datasets/butterflies', teacher_model,
        #                                         student_model,
        #                                         grid_sampling_config,
        #                                         split=split, transform=transform)
        dataset = GridTeacherDataset('nabirds', '../data/nabirds/', teacher_model,
                                                student_model,
                                                grid_sampling_config,
                                                split=split, transform=transform,
                                                dataset_loading_params={'class_list': [i for i in range(0, 20)]})

        print(len(dataset))
        print(dataset[0])
