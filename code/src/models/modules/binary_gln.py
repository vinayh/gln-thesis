import torch
from math import e

# import matplotlib.pyplot as plt

# from matplotlib.animation import FuncAnimation
# import numpy as np

import src.models.modules.rand_halfspace_gln as rand_hspace_gln

from src.utils.helpers import logit, nan_inf_in_tensor
from src.utils.gated_plotter import GatedPlotter


def init_params(layer_sizes, hparams, binary_class=0, X_all=None, y_all=None):
    ctx, W, opt, biases = [], [], [], []
    base_bias = None
    num_contexts = 2 ** hparams["num_subcontexts"]
    use_autograd = hparams["train_autograd_params"]
    p_clip = hparams["pred_clip"]
    # Base bias
    if hparams["base_bias"]:
        base_bias = torch.random.uniform(low=logit(p_clip), high=logit(1 - p_clip))
    # Params for gated layers
    for i in range(1, len(layer_sizes)):
        # input_dim, layer_dim = layer_sizes[i - 1] + 1, layer_sizes[i]
        input_dim, layer_dim = layer_sizes[i - 1], layer_sizes[i]
        layer_ctx = rand_hspace_gln.get_params(hparams, layer_dim)
        layer_W = torch.ones(layer_dim, num_contexts, input_dim) / input_dim
        layer_bias = torch.empty(1, 1).uniform_(logit(p_clip), logit(1 - p_clip))
        # TODO: Currently disabled grad to train on multiple GPUs without
        # error of autograd transferring data across devices
        ctx_param = torch.nn.Parameter(layer_ctx, requires_grad=False)
        W_param = torch.nn.Parameter(layer_W, requires_grad=False)
        bias_param = torch.nn.Parameter(layer_bias, requires_grad=False)
        if hparams["gpu"]:
            ctx_param = ctx_param.cuda()
            W_param = W_param.cuda()
            bias_param = bias_param.cuda()
        # ctx_param = torch.nn.Parameter(layer_ctx, requires_grad=use_autograd)
        layer_opt = (
            torch.optim.SGD(params=[layer_ctx], lr=0.1) if use_autograd else None
        )
        ctx.append(ctx_param)
        W.append(W_param)
        opt.append(layer_opt)
        biases.append(bias_param)
    return {
        "ctx": ctx,
        "weights": W,
        "opt": opt,
        "biases": biases,
        "base_bias": base_bias,
    }


def lr(hparams, t):
    if hparams["dynamic_lr"]:
        return min(hparams["lr"], hparams["lr"] / (1.0 + 1e-3 * t))
    else:
        return hparams["lr"]


def gated_layer(
    params, hparams, logit_x, s, y, l_idx, t, bmap, is_train, use_autograd=False,
):
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
    s = s[:, :-1]  # TODO: bias
    batch_size = s.shape[0]
    layer_dim, _, input_dim = params["weights"][l_idx].shape
    # c: [batch_size, input_dim]
    c = rand_hspace_gln.calc(s, params["ctx"][l_idx], bmap, hparams["gpu"])
    layer_bias = e / (e + 1)
    # TODO: bias
    # logit_x = torch.cat([logit_x, layer_bias * torch.ones_like(logit_x[:, :1])], dim=1)
    if nan_inf_in_tensor(logit_x):
        raise Exception
    # w_ctx: [batch_size, input_dim, output_layer_dim]
    w_ctx = torch.stack(
        [params["weights"][l_idx][range(layer_dim), c[j], :] for j in range(batch_size)]
    )
    if nan_inf_in_tensor(w_ctx):
        raise Exception
    # logit_x_out: [batch_size, output_layer_dim]
    logit_x_out = torch.bmm(w_ctx, logit_x.unsqueeze(2)).flatten(start_dim=1)
    # Clamp to pred_clip
    logit_x_out = torch.clamp(
        logit_x_out,
        min=logit(hparams["pred_clip"]),
        max=logit(1 - hparams["pred_clip"]),
    )
    if nan_inf_in_tensor(logit_x_out):
        raise Exception
    if is_train:
        # loss: [batch_size, output_layer_dim]
        loss = torch.sigmoid(logit_x_out) - y.unsqueeze(1)
        # w_delta: torch.einsum('ab,ac->acb', loss, logit_x)  # [batch_size, input_dim, output_layer_dim]
        w_delta = torch.bmm(loss.unsqueeze(2), logit_x.unsqueeze(1))
        w_new = torch.clamp(
            w_ctx - lr(hparams, t) * w_delta,
            min=-hparams["w_clip"],
            max=hparams["w_clip"],
        )
        if nan_inf_in_tensor(w_new):
            raise Exception
        # [batch_size, input_dim, output_layer_dim]
        with torch.no_grad():
            for j in range(batch_size):
                params["weights"][l_idx][range(layer_dim), c[j], :] = w_new[j]
        #  TODO: Try adding layer_bias here to see if it helps
        if use_autograd:
            # w_ctx: [batch_size, input_dim, output_layer_dim]
            logit_x_out_updated = torch.bmm(w_new, logit_x.unsqueeze(2)).flatten(
                start_dim=1
            )  # [batch_size, output_layer_dim]
            return logit_x_out, params, logit_x_out_updated
    return logit_x_out, params, None


def base_layer(params, hparams, s_bias):
    # TODO: Try using base_bias params to see if it helps
    p_clip = hparams["pred_clip"]
    logit_x_out = logit(torch.clamp(s_bias, min=p_clip, max=1 - p_clip))
    if hparams["base_bias"]:
        return torch.cat([logit_x_out[:, :-1], params["base_bias"]])
    else:
        return logit_x_out[:, :-1]


def add_ctx_to_plot(params, hparams, X_all, y_all, xy, add_to_plot_fn):
    for l_idx in range(hparams["num_layers_used"]):
        Z = rand_hspace_gln.calc_raw(xy, params["ctx"][l_idx], params["bmap"])[:, 0, :]
        for b_idx in range(hparams["num_subctx"]):
            add_to_plot_fn(Z[:, b_idx])

    if hparams["plot"]:
        plotter = GatedPlotter(X_all, y_all, add_ctx_to_plot)
        return plotter


def forward(
    params,
    hparams,
    binary_class,
    t,
    s,
    y,
    bmap,
    is_train=False,
    autograd_fn=None,
    plotter=None,
):
    """Calculate output of Gated Linear Network for input x and side info s

    Args:
        s ([Float * [batch_size, s_dim]]): Batch of side info samples s
        y ([Float * [batch_size]]): Batch of binary targets
        is_train (bool): Whether to train/update on this batch or not (for val/test)
    Returns:
        [Float * [batch_size]]: Batch of GLN outputs (0 < probability < 1)
    """
    use_autograd = hparams["train_autograd_params"]

    def forward_helper(params, s_bias, is_train):
        x = base_layer(params, hparams, s_bias)
        # Gated layers
        for l_idx in range(hparams["num_layers_used"]):
            x, params, x_updated = gated_layer(
                params,
                hparams,
                x,
                s_bias,
                y,
                l_idx,
                t,
                bmap,
                is_train=is_train,
                use_autograd=use_autograd,
            )
            if is_train and use_autograd:
                # TODO: x_updated?
                autograd_fn(x, y, params["opt"][l_idx])
        return x

    s = s.flatten(start_dim=1)
    s_bias = torch.cat([s, torch.ones_like(s[:, :1])], dim=1)
    x = forward_helper(params, s_bias, is_train=is_train)

    # Add frame to animated plot
    if is_train and hparams["plot"]:
        if binary_class == 0 and not (t % 5):
            plotter.save_data(lambda xy: forward_helper(xy, y=None, is_train=False))
    # return torch.sigmoid(x), plotter
    return torch.sigmoid(x)
