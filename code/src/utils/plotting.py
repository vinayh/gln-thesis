import torch
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentError


def gen_xy_grid(x_lim, y_lim, n_x, n_y):
    """Returns grid of XY coordinates to use for calculating output values
        for 2D plots

        Returns:
            [Float * [self.NX, self.NY]]: self.xy
            [Float * [self.NX]]: XX  TODO: Check if type here is correct
            [Float * [self.NY]]: YY  TODO: Check if type here is correct
        """
    with torch.no_grad():
        x_min, x_max = x_lim
        y_min, y_max = y_lim
        xx = np.linspace(x_min, x_max, n_x)
        # Y is from max to min (to plot correctly) instead of min to max
        yy = np.linspace(y_max, y_min, n_y)
        XX, YY = np.meshgrid(xx, yy)
        xy = torch.tensor(
            np.stack([XX.ravel(), YY.ravel(), np.ones(XX.size)]
                     ).T, dtype=torch.float
        )
        return xy, XX, YY


def add_to_plot_fn(ax, XX, YY, hyperplanes):
    """
    Plot decision boundaries (in this case, context function hyperplanes)

    Args:
        boundaries ([type]): [description]

    Returns:
        [type]: [description]
    """
    return


def plot_gated_model(x, y, forward_fn, hyperplane_fn, num_layers, plot_idx=0):
    n_x, n_y = 80, 80
    plot, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="Greens",
               marker=".", edgecolors='k', linewidths=0.5)
    xy, XX, YY = gen_xy_grid(ax.get_xlim(), ax.get_ylim(), n_x, n_y)
    class_index = 0
    for l_idx in range(num_layers):
        # Results of hyperplanes applied to mesh grid
        # hyperplanes: [num_subctx, num_grid_samples]
        hyperplanes = hyperplane_fn(
            xy, l_idx)[:, class_index].squeeze(2).T
        for b in range(hyperplanes.shape[0]):  # For each subctx/branch
            ax.contour(
                XX,
                YY,
                hyperplanes[b].reshape(n_x, n_y),
                colors="k",
                levels=[0],
                alpha=0.5,
                linestyles=["-"],
            )
    # Output of network for each point in mesh grid
    z = torch.stack([forward_fn(xy_i) for xy_i in xy]).squeeze(2)
    z_i = z[:, class_index].reshape(n_x, n_y)
    z_i = (z_i - z_i.min())/(z_i.max() - z_i.min())
    ax.imshow(
        z_i,
        vmin=z_i.min(),
        vmax=z_i.max(),
        cmap="Greens",
        extent=[*ax.get_xlim(), *ax.get_ylim()],
        interpolation="bilinear",
    )
    # plt.show()
    plt.savefig("/home/vinay/gln_plot_frame_{}.png".format(plot_idx))


# def add_ctx_to_plot(xy, add_to_plot_fn):
#     for l_idx in range(hparams["num_layers_used"]):
#         Z = rand_hspace_gln.calc_raw(xy, params["ctx"][l_idx])
#         for b_idx in range(hparams["num_branches"]):
#             add_to_plot_fn(Z[:, b_idx])
