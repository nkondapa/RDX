import os

import torchvision
from mpl_toolkits.axes_grid1 import ImageGrid

from src.nlmcd.source.experiments.eval_utils import *
from src.nlmcd.source.experiments.alignment import *
import matplotlib.pyplot as plt
from PIL import Image
import datetime
from src.nlmcd.executable_scripts import cluster_visualization as cv
import src.utils.plotting_helper as p
import torchvision

def get_nlmcd_selected_indices(config_path, model_ind, exp_dir, date, num_concepts, viz_cfg_path="./source/conf/cluster_visualization.yaml"):

    align_configs, align_results = load_alignment_results(exp_dir, start_date=date)
    align_configs = align_configs.reset_index().set_index("run_id")

    # print(align_configs)

    run_id = 0
    run_dir = align_configs.loc[run_id]["run_dir"]
    heatmap = align_results[run_dir]['alignment']

    # plt.figure()
    # plt.imshow(heatmap, cmap="magma_r")
    # plt.colorbar()
    # plt.show()

    min_alignment_per_cluster = heatmap.T.min(axis=model_ind)
    sorted_inds = np.argsort(min_alignment_per_cluster)
    topk_clusters = sorted_inds[:num_concepts]

    cfg = OmegaConf.load(viz_cfg_path)

    token_idx = None
    soft_assignments = load_concept_activation(
        config_path,
        None,
        train=True,
        cluster_assignment="hdbscan",
        filename_root="clustering.npy",
        take_parent=False,
    )

    hard_assignments = load_concept_activation(
        config_path,
        None,
        train=True,
        cluster_assignment="hard_clustering",
        filename_root="clustering.npy",
        take_parent=False,
    )

    print(soft_assignments.shape, hard_assignments.shape)

    np.random.seed(42)
    selected_indices = []
    for cluster_idx in np.unique(hard_assignments):
        if cluster_idx == -1 or cluster_idx not in topk_clusters:
            continue
        sample_inds = np.where(hard_assignments == cluster_idx)[0]
        sample_k = np.random.choice(sample_inds, size=9, replace=False)
        selected_indices.append(sample_k)
        # images = [transform(Image.open(dataset.dataset.samples[i][0]).convert("RGB")) for i in sample_k]

        # fig, axes = p.make_image_grid(images, mode='3x3')
        # plt.show()

    # where hard assignments are not topk clusters, set to -1
    for idx in range(hard_assignments.shape[0]):
        if hard_assignments[idx] not in topk_clusters:
            hard_assignments[idx] = -1

    return selected_indices, hard_assignments

if __name__ == "__main__":

    folder_list = ['dvd2_gibbons', 'dvd2_mittens', 'dvd2_trolleybus', 'dvd2_whippet', 'clip_vs_clipinat_inat_corvid', 'clip_vs_clipinat_inat_gator', 'clip_vs_clipinat_inat_maple',
                   'dvd2_gibbons_ar10', 'dvd2_mittens_ar10', 'dvd2_trolleybus_ar10', 'dvd2_whippet_ar10', 'clip_vs_clipinat_inat_corvid_ar10', 'clip_vs_clipinat_inat_gator_ar10', 'clip_vs_clipinat_inat_maple_ar10',
                   'dvd2_gibbons_ar01', 'dvd2_mittens_ar01', 'dvd2_trolleybus_ar01', 'dvd2_whippet_ar01', 'clip_vs_clipinat_inat_corvid_ar01', 'clip_vs_clipinat_inat_gator_ar01', 'clip_vs_clipinat_inat_maple_ar01',
                   'dvd2_lp_gibbons', 'dvd2_lp_mittens', 'dvd2_lp_trolleybus', 'dvd2_lp_whippet',
                     'dvd2_mp_gibbons', 'dvd2_mp_mittens', 'dvd2_mp_trolleybus', 'dvd2_mp_whippet'
                   ]

    # folder_list = folder_list[-8:]
    config_dict = {}
    path = './results/concept_discovery'
    exp_dir_path = './results/alignment'
    exp_dirs = []
    for folder in folder_list:
        full_path = os.path.join(path, folder, 'clustering')
        runs = os.listdir(full_path)

        for run in runs:
            if "2025-" in run:
                target_folders = os.listdir(os.path.join(full_path, run))
                config_dict[folder.split('/')[-1]] = tuple([os.path.join(full_path, run, f) for f in target_folders])

        exp_dir_base = os.path.join(exp_dir_path, folder, 'concept_alignment')
        # recurse to bottom folder
        while True:
            subfolders = [f for f in os.listdir(exp_dir_base) if os.path.isdir(os.path.join(exp_dir_base, f))]
            if len(subfolders) == 0:
                break
            exp_dir_base = os.path.join(exp_dir_base, subfolders[0])
        exp_dirs.append(exp_dir_base)


    print(config_dict)
    print(exp_dirs)
    # config_dict = {
    #     'gibbons': ('results/concept_discovery/dvd2_gibbons/clustering/2025-07-27/00-31-59', 'results/concept_discovery/dvd2_gibbons/clustering/2025-07-27/00-32-43'),
    #     'mittens': ('results/concept_discovery/dvd2_mittens/clustering/2025-07-27/00-33-57', 'results/concept_discovery/dvd2_mittens/clustering/2025-07-27/00-34-40'),
    #     'trolleybus': ('results/concept_discovery/dvd2_trolleybus/clustering/2025-07-27/00-35-52', 'results/concept_discovery/dvd2_trolleybus/clustering/2025-07-27/00-36-37'),
    #     'whippet': ('results/concept_discovery/dvd2_whippet/clustering/2025-07-27/00-37-49', 'results/concept_discovery/dvd2_whippet/clustering/2025-07-27/00-38-32'),
    #     'corvid': ('results/concept_discovery/clip_vs_clipinat_inat_corvid/clustering/2025-07-27/01-18-07', 'results/concept_discovery/clip_vs_clipinat_inat_corvid/clustering/2025-07-27/01-18-44'),
    #     'gator': ('results/concept_discovery/clip_vs_clipinat_inat_gator/clustering/2025-07-27/01-19-43', 'results/concept_discovery/clip_vs_clipinat_inat_gator/clustering/2025-07-27/01-20-20'),
    #     'maple': ('results/concept_discovery/clip_vs_clipinat_inat_maple/clustering/2025-07-27/01-21-19', 'results/concept_discovery/clip_vs_clipinat_inat_maple/clustering/2025-07-27/01-22-05'),
    # }

    # exp_dirs = [
    #     "./results/alignment/dvd2_gibbons/concept_alignment/",
    #     "./results/alignment/dvd2_mittens/concept_alignment/",
    #     "./results/alignment/dvd2_trolleybus/concept_alignment/",
    #     "./results/alignment/dvd2_whippet/concept_alignment/",
    #     "./results/alignment/clip_vs_clipinat_inat_corvid/concept_alignment/",
    #     "./results/alignment/clip_vs_clipinat_inat_gator/concept_alignment/",
    #     "./results/alignment/clip_vs_clipinat_inat_maple/concept_alignment/",
    # ]
    out_folders = [
        "dino_vs_dinov2_imagenet/dino_vs_dinov2_imagenet_gibbons",
        "dino_vs_dinov2_imagenet/dino_vs_dinov2_imagenet_mittens",
        "dino_vs_dinov2_imagenet/dino_vs_dinov2_imagenet_trolleybus",
        "dino_vs_dinov2_imagenet/dino_vs_dinov2_imagenet_whippet",
        "clip_vs_clipinat_inat/clip_vs_clipinat_inat_corvid",
        "clip_vs_clipinat_inat/clip_vs_clipinat_inat_gator",
        "clip_vs_clipinat_inat/clip_vs_clipinat_inat_maple",

        "dino_vs_dinov2_imagenet_ar/dino_vs_dinov2_imagenet_ar_gibbons",
        "dino_vs_dinov2_imagenet_ar/dino_vs_dinov2_imagenet_ar_mittens",
        "dino_vs_dinov2_imagenet_ar/dino_vs_dinov2_imagenet_ar_trolleybus",
        "dino_vs_dinov2_imagenet_ar/dino_vs_dinov2_imagenet_ar_whippet",
        "clip_vs_clipinat_inat_ar/clip_vs_clipinat_inat_ar_corvid",
        "clip_vs_clipinat_inat_ar/clip_vs_clipinat_inat_ar_gator",
        "clip_vs_clipinat_inat_ar/clip_vs_clipinat_inat_ar_maple",

        "dino_vs_dinov2_imagenet_ar/dino_vs_dinov2_imagenet_ar_gibbons",
        "dino_vs_dinov2_imagenet_ar/dino_vs_dinov2_imagenet_ar_mittens",
        "dino_vs_dinov2_imagenet_ar/dino_vs_dinov2_imagenet_ar_trolleybus",
        "dino_vs_dinov2_imagenet_ar/dino_vs_dinov2_imagenet_ar_whippet",
        "clip_vs_clipinat_inat_ar/clip_vs_clipinat_inat_ar_corvid",
        "clip_vs_clipinat_inat_ar/clip_vs_clipinat_inat_ar_gator",
        "clip_vs_clipinat_inat_ar/clip_vs_clipinat_inat_ar_maple",

        "dino_vs_dinov2_imagenet_less_points/dino_vs_dinov2_imagenet_less_points_gibbons",
        "dino_vs_dinov2_imagenet_less_points/dino_vs_dinov2_imagenet_less_points_mittens",
        "dino_vs_dinov2_imagenet_less_points/dino_vs_dinov2_imagenet_less_points_trolleybus",
        "dino_vs_dinov2_imagenet_less_points/dino_vs_dinov2_imagenet_less_points_whippet",

        "dino_vs_dinov2_imagenet_more_points/dino_vs_dinov2_imagenet_more_points_gibbons",
        "dino_vs_dinov2_imagenet_more_points/dino_vs_dinov2_imagenet_more_points_mittens",
        "dino_vs_dinov2_imagenet_more_points/dino_vs_dinov2_imagenet_more_points_trolleybus",
        "dino_vs_dinov2_imagenet_more_points/dino_vs_dinov2_imagenet_more_points_whippet",
    ]
    dset_names = [
        "imagenet_subset_grouped",
        "imagenet_subset_grouped",
        "imagenet_subset_grouped",
        "imagenet_subset_grouped",
        "inatdl_subset_grouped",
        "inatdl_subset_grouped",
        "inatdl_subset_grouped"
    ]
    method_names = ["nlmcd"] * len(dset_names) + ["nlmcd_ar1to0"] * len(dset_names) + ["nlmcd_ar0to1"] * len(dset_names)
    dset_names = dset_names * 3

    method_names.extend(["nlmcd"] * 8)
    dset_names.extend(["imagenet_subset_grouped"] * 8)

    # method_names = method_names[-8:]
    # dset_names = dset_names[-8:]

    date = "2024-10-11_15-00-00"
    # date = "2025-09-18_13-00-00"

    output_root = f'/home/nkondapa/PycharmProjects/ConceptDiff/outputs2/'
    for i, (config_id, config_paths) in enumerate(config_dict.items()):
        method_name = method_names[i]
        dset_name = dset_names[i]
        out_path = os.path.join(output_root, out_folders[i], dset_name, method_name)
        print(out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        sel_inds_0, hard_a0 = get_nlmcd_selected_indices(config_paths[0], 0, exp_dirs[i], date=date, viz_cfg_path="./source/conf/cluster_visualization.yaml")
        sel_inds_1, hard_a1 = get_nlmcd_selected_indices(config_paths[1], 1, exp_dirs[i], date=date,  viz_cfg_path="./source/conf/cluster_visualization.yaml")

        d = {"0": {"selected_indices": sel_inds_0},
             "1": {"selected_indices": sel_inds_1}}

        with open(os.path.join(out_path, 'fig_paths.pkl'), 'wb') as f:
            pickle.dump(d, f)

        with open(os.path.join(out_path, 'outputs.pkl'), 'wb') as f:
            pickle.dump({"inputs": {"add_null_cluster": False}, "repr_0": hard_a0, "repr_1": hard_a1}, f)
