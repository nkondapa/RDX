from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import numpy as np



def shrink_cbar(ax, shrink=0.9):
    b = ax.get_position()
    new_h = b.height*shrink
    pad = (b.height-new_h)/2.
    new_y0 = b.y0 + pad
    new_y1 = b.y1 - pad
    b.y0 = new_y0
    b.y1 = new_y1
    ax.set_position(b)


def make_axes_invisible(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def make_image_grid(sel_images, mode='3x3', axes_pad=0.25):
    if mode == '2x5':
        fig = plt.figure(figsize=(14.5, 5.95))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 5),  # creates 2x2 grid of Axes
                         axes_pad=axes_pad,  # pad between Axes in inch.
                         )
    elif mode == '3x3':
        fig = plt.figure(figsize=(9., 9.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(3, 3),  # creates 2x2 grid of Axes
                         axes_pad=axes_pad,  # pad between Axes in inch.
                         )
    elif mode == '4x4':
        fig = plt.figure(figsize=(12., 12.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(4, 4),  # creates 2x2 grid of Axes
                         axes_pad=axes_pad,  # pad between Axes in inch.
                         )
    elif mode == '5x5':
        fig = plt.figure(figsize=(15., 15.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(5, 5),  # creates 2x2 grid of Axes
                         axes_pad=axes_pad,  # pad between Axes in inch.
                         )
    elif mode == '3x4':
        fig = plt.figure(figsize=(12., 9.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(3, 4),  # creates 2x2 grid of Axes
                         axes_pad=axes_pad,  # pad between Axes in inch.
                         )
    elif mode == '4x3':
        fig = plt.figure(figsize=(9., 12.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(4, 1),  # creates 2x2 grid of Axes
                         axes_pad=axes_pad,  # pad between Axes in inch.
                         )

    for i, ax in enumerate(grid):
        # Iterating over the grid returns the Axes.
        if i < len(sel_images):
            im = sel_images[i]
            ax.imshow(im)
        make_axes_invisible(ax)

    # plt.tight_layout()
    return fig, grid


def finish_plot(show, save, save_path=None, fig=None):
    if save:
        if fig is None:
            plt.savefig(save_path)
        else:
            fig.savefig(save_path)

    if show:
        plt.show()

    if fig is not None:
        plt.close(fig)
    else:
        plt.close('all')


def make_large_cmap(num_clusters):
    get_marker = lambda x: 'o'
    legend_plot = None

    if 10 < num_clusters < 120:
        cmaps = [plt.get_cmap('tab20'), plt.get_cmap('tab20b'), plt.get_cmap('Dark2'), plt.get_cmap('Set3')] * 2
        lens = [20, 20, 8, 12] * 2
        cum_lens = np.cumsum(lens)

        def cmap(x_list):
            out = []
            for x in x_list:
                for i, cl in enumerate(cum_lens):
                    if x < cl:
                        out.append(cmaps[i](x % lens[i]))
                        break
            return out

        def get_marker(x):
            marker_opts = ['o', 's', 'v', '^', 'D', 'P', 'X', 'H', 'd', 'p', 'x', 'h', '8', '*', '+', '1', '2', '3',
                           '4', '8']
            for i, cl in enumerate(cum_lens):
                if x < cl:
                    return marker_opts[i]

        legend_plot = None
        if num_clusters > 25:
            # create legend plot
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(12, 4)
            for i in range(num_clusters):
                ax.scatter(i, 0.2, c=cmap([i]), label=f'{i}', marker=get_marker(i))
            ax.set_ylim(-0.01, 1)
            ax.legend(ncol=10)
            plt.tight_layout()
            legend_plot = fig

    elif num_clusters <= 10:
        cmap_base = plt.get_cmap('tab10')
        cmap_sec = plt.get_cmap('Dark2')
        # cmap = cmap_base

        def cmap(x_list):
            out = []
            if x_list[0] == 7:
                return cmap_sec([0] * len(x_list))
            else:
                return cmap_base(x_list)

        def get_marker(x):
            return 'o'

    elif num_clusters >= 60:
        _cmap = plt.get_cmap('magma')
        cmap = lambda x: _cmap(x / num_clusters)

    return cmap, get_marker, legend_plot