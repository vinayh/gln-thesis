import torch
from math import e
# import matplotlib.pyplot as plt

# from matplotlib.animation import FuncAnimation
# import numpy as np

import src.models.modules.rand_halfspace_gln as rand_hspace_gln
# from src.utils.helpers import logit
from src.utils.gated_plotter import GatedPlotter


def init_params(num_neurons, hparams, binary_class=0, X_all=None, y_all=None):
    ctx, W, opt = [], [], []
    num_contexts = 2**hparams["num_subcontexts"]
    bitwise_map = torch.tensor(
        [2**i for i in range(hparams["num_subcontexts"])])
    for i in range(1, len(num_neurons)):
        with torch.no_grad():
            input_dim, layer_dim = num_neurons[i-1], num_neurons[i]
            layer_ctx = rand_hspace_gln.get_params(hparams["input_size"] + 1,
                                                   layer_dim,
                                                   hparams["num_subcontexts"])
            layer_W = 0.5 * \
                torch.ones(
                    layer_dim, num_contexts, input_dim + 1)
            ctx_param = torch.nn.Parameter(layer_ctx, requires_grad=True)
            W_param = torch.nn.Parameter(layer_W, requires_grad=True)
        if hparams["gpu"]:
            ctx_param = ctx_param.cuda()
            W_param = W_param.cuda()
        layer_opt = torch.optim.SGD(params=[ctx_param], lr=0.1)
        ctx.append(ctx_param)
        W.append(W_param)
        opt.append(layer_opt)
    return {"ctx": ctx, "weights": W, "opt": opt, "bmap": bitwise_map}


def lr(hparams, t):
    return 0.1
    # return min(0.1, 0.2/(1.0 + 1e-2 * t))


def gated_layer(params, hparams, logit_x, s, y, l_idx, t, is_train, is_gpu=False,
                use_autograd=False):
    """Using provided input activations, context functions, and weights,
       returns the result of the GLN layer

    Args:
        logit_x ([Float * [batch_size, input_dim]]): Input activations (with logit applied)
        s ([Float * [batch_size, s_dim]]): Batch of side info samples s
        y ([Float * [batch_size]]): Batch of binary targets
        l_idx (Int): Index of layer to use for selecting correct ctx and layer weights
        is_train (bool): Whether to train/update on this batch or not (for val/test)
    Returns:
        [Float * [batch_size, output_layer_dim]]: Output of GLN layer
    """
    batch_size = s.shape[0]
    layer_dim, _, input_dim = params["weights"][l_idx].shape
    # c: [batch_size, input_dim]
    c = rand_hspace_gln.calc(s, params["ctx"][l_idx],
                             params["bmap"], hparams["gpu"])
    layer_bias = e / (e+1)
    logit_x = torch.cat([logit_x,
                         layer_bias * torch.ones(
                             logit_x.shape[0], 1, device=hparams["device"])],
                        dim=1)
    # w_ctx: [batch_size, input_dim, output_layer_dim]
    w_ctx = torch.stack([params["weights"][l_idx][range(layer_dim), c[j], :]
                         for j in range(batch_size)])
    x_out = torch.bmm(w_ctx, logit_x.unsqueeze(2)).flatten(
        start_dim=1)  # [batch_size, output_layer_dim]
    if is_train:
        # loss: [batch_size, output_layer_dim]
        loss = torch.sigmoid(x_out) - y.unsqueeze(1)
        # w_delta: torch.einsum('ab,ac->acb', loss, logit_x)  # [batch_size, input_dim, output_layer_dim]
        w_delta = torch.bmm(loss.unsqueeze(2), logit_x.unsqueeze(1))
        w_new = torch.clamp(w_ctx - lr(hparams, t) * w_delta,
                            min=-hparams["w_clip"], max=hparams["w_clip"])
        # [batch_size, input_dim, output_layer_dim]
        with torch.no_grad():
            for j in range(batch_size):
                params["weights"][l_idx][range(layer_dim), c[j], :] = w_new[j]
        if use_autograd:
            # w_ctx: [batch_size, input_dim, output_layer_dim]
            w_ctx_updated = torch.stack([params["weights"][l_idx][range(layer_dim), c[j], :]
                                         for j in range(batch_size)])
            x_out_updated = torch.bmm(w_ctx_updated, logit_x.unsqueeze(2)).flatten(
                start_dim=1)  # [batch_size, output_layer_dim]
            return x_out, params, x_out_updated
    return x_out, params, None


def base_layer(s_bias, layer_size):
    return s_bias[:, :-1]


def add_ctx_to_plot(params, hparams, X_all, y_all, xy, add_to_plot_fn):
    for l_idx in range(hparams["num_layers_used"]):
        Z = rand_hspace_gln.calc_raw(xy, params["ctx"][l_idx],
                                     params["bmap"])[:, 0, :]
        for b_idx in range(hparams["num_subctx"]):
            add_to_plot_fn(Z[:, b_idx])

    if hparams["plot"]:
        plotter = GatedPlotter(X_all, y_all, add_ctx_to_plot)
        return plotter


def forward(params, hparams, binary_class, t, s, y, is_train=False,
            use_autograd=False, autograd_fn=None, plotter=None):
    """Calculate output of Gated Linear Network for input x and side info s

    Args:
        s ([Float * [batch_size, s_dim]]): Batch of side info samples s
        y ([Float * [batch_size]]): Batch of binary targets
        is_train (bool): Whether to train/update on this batch or not (for val/test)
    Returns:
        [Float * [batch_size]]: Batch of GLN outputs (0 < probability < 1)
    """
    def forward_helper(params, s_bias, is_train):
        x = base_layer(s_bias, hparams["input_size"])
        # Gated layers
        for l_idx in range(hparams["num_layers_used"]):
            x, params, x_updated = gated_layer(params, hparams, x, s_bias,
                                               y, l_idx, t, is_train=is_train,
                                               is_gpu=False,
                                               use_autograd=use_autograd)
            if is_train and use_autograd:
                autograd_fn(x_updated, y, params["opt"][l_idx])
        return x

    s = s.flatten(start_dim=1)
    s_bias = torch.cat(
        [s, torch.ones(s.shape[0], 1, device=hparams["device"])], dim=1)
    x = forward_helper(params, s_bias, is_train=is_train)

    # Add frame to animated plot
    if is_train and hparams["plot"]:
        if binary_class == 0 and not (hparams["t"] % 5):
            plotter.save_data(
                lambda xy: forward_helper(xy, y=None, is_train=False))
    # return torch.sigmoid(x), plotter
    return torch.sigmoid(x)

# def base_layer_old(params, hparams, s, y, layer_size):
#     batch_size = s.shape[0]
#     clip_param = hparams["pred_clipping"]
#     rand_activations = torch.empty(batch_size, hparams["l_sizes"][0],
#                                    device=hparams["device"]).normal_(
#                                        mean=0.5, std=1.0)
#     # rand_activations.requires_grad = False
#     # x = self.W_base(rand_activations)
#     # x = torch.clamp(x, min=clip_param, max=1.0-clip_param)
#     x = torch.clamp(rand_activations, min=clip_param, max=1.0-clip_param)
#     x = logit(x)
#     return torch.ones(batch_size, layer_size, device=hparams["device"])/2.0


# def base_layer_old2(s_bias, layer_size):
    # Weights for base predictor (arbitrary activations)
    # self.W_base = torch.nn.Linear(self.l_sizes[0], self.l_sizes[0])
    # self.W_base.requires_grad_ = False
    # torch.nn.init.normal_(self.W_base.weight.data, mean=0.0, std=0.2)
    # torch.nn.init.normal_(self.W_base.bias.data, mean=0.0, std=0.2)

    # mean = torch.mean(s, dim=0)
    # stdev = torch.std(s, dim=0)
    # x = (s - mean) / (stdev + 1.0)
