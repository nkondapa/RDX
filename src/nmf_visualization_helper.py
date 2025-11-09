import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils import plotting_helper as ph
from PIL import Image
from matplotlib import colors
import torchvision


def generate_visualizations(input_dict, output_dict, plot_params=None):
    output_folder = output_dict.get('method_dir', None)
    viz_output_base_folder = f'{output_folder}/visualizations'

    plot_params = {} if plot_params is None else plot_params
    fig_paths = {}
    fontsize = plot_params.get('fontsize', 12)
    show, save = plot_params.get('show', False), plot_params.get('save', False)

    r0_red = input_dict['red0']
    r1_red = input_dict['red1']
    dataset_labels = input_dict['dataset_labels']
    representations = input_dict['representations']
    image_samples = input_dict['image_samples']
    image_samples_is_paths = type(image_samples[0]) == np.str_
    basic_transform = torchvision.transforms.Resize((224, 224))

    r0_dm = torch.cdist(representations[0], representations[0])
    r1_dm = torch.cdist(representations[1], representations[1])
    num_clusters = input_dict.get("num_concepts_for_user", input_dict.get('n_components', input_dict.get('n_clusters', None)))
    cluster_list = list(range(num_clusters))

    names_dirs = ['Model 0', 'Model 1']
    dirs = ['0', '1']
    for di, d in enumerate(dirs):
        fig_paths[d] = {}
        viz_output_folder = os.path.join(viz_output_base_folder, str(d))
        os.makedirs(viz_output_folder, exist_ok=True)
        name = names_dirs[di]

        out_data = output_dict[f'repr_{d}']
        U, V = out_data['U'], out_data['V']
        cmap, get_marker, legend_plot = ph.make_large_cmap(len(cluster_list))

        if plot_params.get('visualize_coeffs', False):
            fig, axes = plt.subplots(1, 1, squeeze=False)
            axes[0, 0].imshow(U, cmap='magma', aspect='auto', interpolation='nearest')
            axes[0, 0].set_title(f'{name} Coefficients', fontsize=fontsize)
            axes[0, 0].set_xlabel('Components', fontsize=fontsize)
            axes[0, 0].set_ylabel('Inputs', fontsize=fontsize)
            # colorbar
            cbar = plt.colorbar(axes[0, 0].imshow(U, cmap='magma', aspect='auto', interpolation='nearest'), ax=axes[0, 0])
            cbar.set_label('Coefficient Value', fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)
            if save:
                plt.savefig(f'{viz_output_base_folder}/coeffs_{d}.png', bbox_inches='tight')
                fig_paths[d][f'coeffs_{di}'] = f'{viz_output_base_folder}/coeffs_{d}.png'
            if show:
                plt.show()


        ##### CLUSTER FIGURE START #####
        # subsample from cluster to generate grids
        grid_size = plot_params.get('grid_size', '4x4')
        indices = np.arange(0, r0_red.shape[0])
        num_samples = plot_params.get('num_samples', 16)
        explanations_folder = f'{viz_output_folder}/explanations'
        os.makedirs(explanations_folder, exist_ok=True)
        fig_paths[d]['clusters'] = []
        selected_indices = []
        for i in cluster_list:
            sel_inds = U[:, i].argsort()[::-1][:num_samples]
            if image_samples_is_paths:
                sel_paths = image_samples[sel_inds]
                sel_images = [basic_transform(Image.open(p)) for p in sel_paths]
            else:
                sel_images = image_samples[sel_inds.tolist()].permute(0, 2, 3, 1).cpu().numpy()
            fig, grid = ph.make_image_grid(sel_images, mode=grid_size, axes_pad=0.3)
            # plt.suptitle(f'Cluster {i}', fontsize=fontsize)
            plt.tight_layout()
            # plt.show()
            fig_path = f'{explanations_folder}/{i}.png'
            ph.finish_plot(show, save, save_path=fig_path, fig=fig)
            fig_paths[d]['clusters'].append(fig_path)
            selected_indices.append(sel_inds.tolist())
        fig_paths[d]['selected_indices'] = selected_indices

        # summarize cluster figures in a single figure
        pad = plot_params.get('cl_summ_pad', 10)
        summ_size = 600
        summ_figs = fig_paths[d]['clusters']
        bg_color = plot_params.get('cl_summ_bg_color', (0, 0, 0))
        bg_color = tuple(bg_color) if type(bg_color) == list else bg_color

        if len(summ_figs) != 0:
            cl_summ_im = Image.new('RGB', (len(summ_figs) * (summ_size + pad) + pad, summ_size + 2 * pad), color=bg_color)
            for i, fig_path in enumerate(summ_figs):
                im = Image.open(fig_path)
                im = im.resize((summ_size, summ_size))
                cl_summ_im.paste(im, (i * (summ_size + pad) + pad, pad))
            if show:
                fig = plt.figure()
                fig.set_size_inches(20, 3)
                plt.imshow(cl_summ_im)
                plt.show()
            if save:
                cl_summ_im.save(f'{viz_output_base_folder}/cluster_summ_{d}.png')
                fig_paths[d]['cluster_summ'] = f'{viz_output_base_folder}/cluster_summ_{d}.png'
        ##### CLUSTER FIGURE END #####

        ##### SELECTED CLUSTER FIGURE START #####
        fig, axes = plt.subplots(1, 2, squeeze=False)
        fig.set_size_inches(12, 6)

        axes[0, 0].scatter(r0_red[:, 0], r0_red[:, 1], c='gray', alpha=0.2)
        axes[0, 1].scatter(r1_red[:, 0], r1_red[:, 1], c='gray', alpha=0.2)

        for cli in range(len(selected_indices)):
            curr_si = selected_indices[cli]
            c = cmap([cli])
            alpha = 1
            marker = get_marker(cli)
            axes[0, 0].scatter(r0_red[curr_si, 0], r0_red[curr_si, 1], color=c, alpha=alpha, marker=marker, label=f'{cli}')
            axes[0, 1].scatter(r1_red[curr_si, 0], r1_red[curr_si, 1], color=c, alpha=alpha, marker=marker)

        axes[0, 0].set_title(f'{names_dirs[0]}', fontsize=fontsize)
        axes[0, 1].set_title(f'{names_dirs[1]}', fontsize=fontsize)
        axes[0, 0].set_xlabel('PC 1', fontsize=fontsize)
        axes[0, 0].set_ylabel('PC 2', fontsize=fontsize)
        axes[0, 1].set_xlabel('PC 1', fontsize=fontsize)
        axes[0, 1].set_ylabel('PC 2', fontsize=fontsize)
        if len(selected_indices) < 35:
            axes[0, 0].legend()
        plt.tight_layout()
        fig_path = f'{viz_output_folder}/sel_inds_viz.png'
        ph.finish_plot(show, save, save_path=fig_path, fig=fig)
        fig_paths[d]['sel_inds_viz'] = fig_path

        ##### SELECTED CLUSTER FIGURE END #####
    return fig_paths