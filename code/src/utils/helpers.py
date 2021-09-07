import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentError
import torch


def entropy(p):
    if p == 0.0 or p == 1.0 or p.isnan():
        return torch.tensor(0)
    return p * torch.log(p) + (1 - p) * torch.log(1 - p)


def logit(x):
    if torch.is_tensor(x):
        return torch.log(x / (1 - x))
    else:
        return torch.log(torch.tensor(x / (1 - x)))


def logit_geo_mix(logit_prev_layer, weights):
    """Geometric weighted mixture of prev_layer using weights

    Args:
        logit_prev_layer ([Float] * n_neurons): Logit of prev layer activations
        weights ([Float] * n_neurons): Weights for neurons in prev layer

    Returns:
        [Float]: sigmoid(w * logit(prev_layer)), i.e. output of current neuron
    """
    return torch.sigmoid(weights.matmul(logit_prev_layer))


def to_one_vs_all(targets, num_classes, device="cpu"):
    """[summary]

    Args:
        targets ([type]): [description]
        num_classes ([type]): [description]
        device (str, optional): [description]. Defaults to 'cpu'.

    Returns:
        [type]: [description]
    """
    ova_targets = torch.zeros(
        (num_classes, len(targets)), dtype=torch.int, device=device
    )
    for i in range(num_classes):
        ova_targets[i, :][targets == i] = 1
    return ova_targets


# class STEFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return (input > 0).bool()

#     @staticmethod
#     def backward(ctx, grad_output):
#         return torch.nn.functional.hardtanh(grad_output)


# class StraightThroughEstimator(torch.nn.Module):
#     def __init__(self):
#         super(StraightThroughEstimator, self).__init__()

#     def forward(self, x):
#         x = STEFunction.apply(x)
#         return x


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.heaviside(
            input, torch.zeros(1).type_as(input)
        ).float()  # this outputs 0 or 1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def nan_inf_in_tensor(x):
    return torch.sum(torch.isinf(x)) + torch.sum(torch.isnan(x)) > 0


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
            np.stack([XX.ravel(), YY.ravel(), np.ones(XX.size)]).T, dtype=torch.float
        )
        return xy, XX, YY


def plot_gated_model(x, y, forward_fn):
    n_x, n_y = 60, 60
    plot, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="hot", marker=".", linewidths=0.0)
    xy, XX, YY = gen_xy_grid(ax.get_xlim(), ax.get_ylim(), n_x, n_y)
    z = torch.stack([forward_fn(xy_i) for xy_i in xy]).squeeze(2)
    class_index = 0
    ax.imshow(
        z[:, class_index].reshape(n_x, n_y),
        vmin=z.min(),
        vmax=z.max(),
        cmap="viridis",
        extent=[*ax.get_xlim(), *ax.get_ylim()],
        interpolation="none",
    )
    plt.savefig("test.png")


# def add_ctx_to_plot(xy, add_to_plot_fn):
#     for l_idx in range(hparams["num_layers_used"]):
#         Z = rand_hspace_gln.calc_raw(xy, params["ctx"][l_idx])
#         for b_idx in range(hparams["num_branches"]):
#             add_to_plot_fn(Z[:, b_idx])
