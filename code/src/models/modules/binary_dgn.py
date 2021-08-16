import torch
from math import e

import src.models.modules.rand_halfspace_dgn as rand_hspace_dgn
from src.utils.helpers import nan_inf_in_tensor, inv_sigmoid

# from src.utils.gated_plotter import GatedPlotter

LAYER_BIAS = e / (e + 1)


def init_params(layer_sizes, hparams, binary_class=0, X_all=None, y_all=None):
    ctx, W, opt = [], [], []
    use_autograd = hparams["train_autograd_params"]
    for i in range(1, len(layer_sizes)):
        input_dim, layer_dim = layer_sizes[i - 1], layer_sizes[i]
        layer_ctx = rand_hspace_dgn.get_params(hparams, layer_dim)
        layer_W = 0.5 * torch.ones(
            layer_dim, hparams["num_branches"], input_dim + 1, device=hparams.device
        )
        ctx_param = torch.nn.Parameter(layer_ctx, requires_grad=use_autograd)
        W_param = torch.nn.Parameter(layer_W, requires_grad=False)
        if hparams["gpu"]:
            W_param = W_param.cuda()
        layer_opt = (
            torch.optim.SGD(params=[ctx_param], lr=0.1) if use_autograd else None
        )
        ctx.append(ctx_param)
        W.append(W_param)
        opt.append(layer_opt)
    return {"ctx": ctx, "weights": W, "opt": opt}


def lr(hparams, t):
    if hparams["dynamic_lr"]:
        return min(hparams["lr"], hparams["lr"] / (1.0 + 1e-3 * t))
    else:
        return hparams["lr"]


def gated_layer(params, hparams, h, s, y, l_idx, t, is_train, use_autograd=False):
    """Using provided input activations, context functions, and weights,
        returns the result of the DGN layer

    Args:
        h ([Float * [batch_size, input_dim]]): Input activations with logit
        s ([Float * [batch_size, self.s_dim]]): Batch of side info samples s
        y ([Float * [batch_size]]): Batch of binary targets
        l_idx (Int): Layer index for selecting ctx and layer weights
        is_train (bool): Whether to train/update on this batch or not
    Returns:
        [Float * [batch_size, layer_dim]]: Output of DGN layer
    """
    h = torch.cat([h, LAYER_BIAS * torch.ones_like(h[:, :1])], dim=1)
    # c: [batch_size, layer_dim]
    c = rand_hspace_dgn.calc(s, params["ctx"][l_idx], hparams.gpu)
    # weights: [batch_size, layer_dim, input_dim]
    weights = torch.bmm(c.float().permute(2, 0, 1), params["weights"][l_idx]).permute(
        1, 0, 2
    )
    # h_out: [batch_size, layer_dim]
    h_out = torch.bmm(weights, h.unsqueeze(2)).squeeze(2)
    if is_train:
        targets = y.unsqueeze(1)
        r_out = torch.sigmoid(h_out)
        r_out_clipped = torch.clamp(
            r_out, min=hparams["pred_clip"], max=1 - hparams["pred_clip"]
        )
        # learn_gates: [batch_size, layer_dim]
        learn_gates = (torch.abs(targets - r_out) > hparams["pred_clip"]).float()
        w_grad1 = (r_out_clipped - targets) * learn_gates
        w_grad2 = torch.bmm(w_grad1.unsqueeze(2), h.unsqueeze(1))
        w_delta = torch.bmm(c.float().permute(2, 1, 0), w_grad2.permute(1, 0, 2))
        # assert(w_delta.shape == W[l_idx].shape)
        # TODO delete this: W.grad = w_delta
        with torch.no_grad():
            params["weights"][l_idx] = (
                params["weights"][l_idx] - lr(hparams, t) * w_delta
            )
        # Get new layer output with updated weights
        if use_autograd:
            # weights: [batch_size, layer_dim, input_dim]
            weights = torch.bmm(
                c.float().permute(2, 0, 1), params["weights"][l_idx]
            ).permute(1, 0, 2)
            # h_out_updated: [batch_size, layer_dim]
            h_out_updated = torch.bmm(weights, h.unsqueeze(2)).squeeze(2)
            return h_out, params, h_out_updated
    return h_out, params, None


def gated_layer2(params, hparams, r, s, y, l_idx, t, is_train, use_autograd=False):
    """Using provided input activations, context functions, and weights,
        returns the result of the DGN layer

    Args:
        r ([Float * [batch_size, input_dim]]): Input activations with sigmoid
        s ([Float * [batch_size, self.s_dim]]): Batch of side info samples s
        y ([Float * [batch_size]]): Batch of binary targets
        l_idx (Int): Layer index for selecting ctx and layer weights
        is_train (bool): Whether to train/update on this batch or not
    Returns:
        [Float * [batch_size, layer_dim]]: Output of DGN layer
    """
    curr_lr = lr(hparams, t)
    # h = torch.cat([h, LAYER_BIAS * torch.ones_like(h[:, :1])], dim=1)
    r_in = torch.cat([r, torch.sigmoid(torch.ones_like(r[:, :1]))], dim=1)
    h_in = inv_sigmoid(r_in)
    # c: [batch_size, layer_dim]
    c = rand_hspace_dgn.calc(s, params["ctx"][l_idx], hparams.gpu)
    # weights: [batch_size, layer_dim, input_dim]
    weights = torch.bmm(c.float().permute(2, 0, 1), params["weights"][l_idx]).permute(
        1, 0, 2
    )
    # h_out: [batch_size, layer_dim]
    h_out = torch.bmm(weights, h_in.unsqueeze(2)).squeeze(2)
    r_out_unclipped = torch.sigmoid(h_out)
    r_out = torch.clamp(
        r_out_unclipped, min=hparams["pred_clip"], max=1 - hparams["pred_clip"]
    )
    if is_train:
        targets = y.unsqueeze(1)
        # learn_gates: [batch_size, layer_dim]
        learn_gates = torch.abs(targets - r_out) > hparams["pred_clip"]
        # w_grad1: [batch_size, layer_dim]
        w_grad1 = (r_out - targets) * learn_gates
        if nan_inf_in_tensor(w_grad1):
            raise Exception
        w_grad2 = torch.bmm(w_grad1.unsqueeze(2), h_in.unsqueeze(1))
        if nan_inf_in_tensor(w_grad2):
            raise Exception
        w_delta = torch.bmm(c.float().permute(2, 1, 0), w_grad2.permute(1, 0, 2))
        if nan_inf_in_tensor(w_delta):
            raise Exception
        # assert(w_delta.shape == W[l_idx].shape)
        # TODO delete this: W.grad = w_delta
        # optimizer[].step(0)
        with torch.no_grad():
            params["weights"][l_idx] = params["weights"][l_idx] - curr_lr * w_delta
        # Get new layer output with updated weights

        if use_autograd:
            # weights: [batch_size, layer_dim, input_dim]
            weights_updated = torch.bmm(
                c.float().permute(2, 0, 1), params["weights"][l_idx]
            ).permute(1, 0, 2)
            # h_out_updated: [batch_size, layer_dim]
            r_out_updated = torch.sigmoid(
                torch.bmm(weights_updated, h_in.unsqueeze(2)).squeeze(2)
            )
            return r_out.detach(), params, r_out_updated
    return r_out.detach(), params, None


def base_layer(s_bias, hparams):
    return s_bias[:, :-1]


def base_layer2(s_bias, hparams):
    return torch.clamp(
        torch.sigmoid(s_bias[:, :-1]),
        min=hparams["pred_clip"],
        max=1 - hparams["pred_clip"],
    ).detach()


def forward(
    params,
    hparams,
    binary_class,
    t,
    s,
    y,
    is_train=False,
    autograd_fn=None,
    plotter=None,
):
    """Calculate output of DGN for input x and side info s

    Args:
        s ([Float * [batch_size, self.s_dim]]): Batch of side info samples s
        y ([Float * [batch_size]]): Batch of binary targets
        is_train (bool): Whether to train/update on this batch or not
    Returns:
        [Float * [batch_size]]: Batch of DGN outputs (0 < probability < 1)
    """
    use_autograd = hparams["train_autograd_params"]

    def forward_helper(params, s_bias, is_train):
        r = base_layer2(s_bias, hparams)
        # Gated layers
        for l_idx in range(hparams["num_layers_used"]):
            r, params, r_updated = gated_layer2(
                params,
                hparams,
                r,
                s_bias,
                y,
                l_idx,
                t,
                is_train=is_train,
                use_autograd=use_autograd,
            )
            if is_train and use_autograd:
                autograd_fn(r_updated, y, params["opt"][l_idx])
        return r

    if len(s.shape) > 1:
        s = s.flatten(start_dim=1)
    s_bias = torch.cat([s, torch.ones(s.shape[0], 1, device=hparams.device)], dim=1)
    r = forward_helper(params, s_bias, is_train=is_train)
    # Add frame to animated plot
    if is_train and hparams["plot"]:
        if binary_class == 0 and not (t % 5):
            plotter.save_data(lambda xy: forward_helper(xy, y=None, is_train=False))
    # return r, plotter
    return r


# def configure_optimizers(params):  # Global straight-through, NO neuron updates
#     return [torch.optim.SGD(params=params["ctx"], lr=0.1)]
#     # return [torch.optim.Adam(params=params["ctx"], lr=0.1)]
