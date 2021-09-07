import torch

import src.models.modules.rand_halfspace_dgn as rand_hspace_dgn

from src.utils.helpers import nan_inf_in_tensor, logit

L1_LOSS_FN = torch.nn.L1Loss(reduction="sum")

# from src.utils.gated_plotter import GatedPlotter


def lr(hparams, t):
    if hparams["dynamic_lr"]:
        return min(hparams["lr"], hparams["lr"] / (1.0 + 1e-3 * t))
    else:
        return hparams["lr"]


def forward_helper(
    params, hparams, s_bias, t, autograd_fn=None, targets=None, is_train=False,
):
    h = base_layer(s_bias, hparams)
    # Gated layers
    for l_idx in range(hparams["num_layers_used"]):
        h, params, h_updated = gated_layer(
            params, hparams, h, s_bias, targets, l_idx, t, is_train=is_train
        )
        # if is_train and hparams.autograd_local and not hparams.autograd_local_w:
        #     r_updated = torch.sigmoid(h_updated)
        #     autograd_fn(r_updated, targets, params["opt_l"][l_idx])
    if is_train and hparams.autograd_global:
        loss = L1_LOSS_FN(torch.sigmoid(h), targets)
        params["opt_g"].zero_grad()
        loss.backward()
        params["opt_g"].step()
    return torch.sigmoid(h.detach())


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
    if len(s.shape) > 1:
        s = s.flatten(start_dim=1)
    s_bias = torch.cat([s, torch.ones_like(s[:, :1])], dim=1)
    r = forward_helper(
        params,
        hparams,
        s_bias,
        t,
        autograd_fn=autograd_fn,
        targets=y.unsqueeze(1),
        is_train=is_train,
    )
    # Add frame to animated plot
    if is_train and hparams["plot"]:
        if binary_class == 0 and not (t % 5):
            plotter.save_data(lambda xy: forward_helper(xy, y=None, is_train=False))
    # return r, plotter
    return r


def gated_layer(params, hparams, h_in, s, targets, l_idx, t, is_train):
    """Using provided input activations, context functions, and weights,
        returns the result of the DGN layer

    Args:
        h_in ([Float * [batch_size, input_dim]]): Input activations with logit
        s ([Float * [batch_size, self.s_dim]]): Batch of side info samples s
        y ([Float * [batch_size]]): Batch of binary targets
        l_idx (Int): Layer index for selecting ctx and layer weights
        is_train (bool): Whether to train/update on this batch or not
    Returns:
        [Float * [batch_size, layer_dim]]: Output of DGN layer
    """
    P_CLIP = hparams["pred_clip"]
    h_in = torch.cat([h_in, torch.ones_like(h_in[:, :1])], dim=1)
    # c: [batch_size, layer_dim]
    c = rand_hspace_dgn.calc(s, params["ctx"][l_idx])
    # weights: [batch_size, layer_dim, input_dim]
    weights = torch.bmm(c.permute(2, 0, 1), params["W"][l_idx]).permute(1, 0, 2)
    # h_out: [batch_size, layer_dim]
    h_out = torch.bmm(weights, h_in.unsqueeze(2)).squeeze(2)
    h_out_updated = None
    if is_train:
        r_out_unclipped = torch.sigmoid(h_out)
        r_out = torch.clamp(r_out_unclipped, min=P_CLIP, max=1 - P_CLIP)
        # learn_gates: [batch_size, layer_dim]
        learn_gates = (torch.abs(targets - r_out_unclipped) > P_CLIP).float()
        w_grad = torch.bmm(
            ((r_out - targets) * learn_gates).unsqueeze(2), h_in.unsqueeze(1)
        )
        w_delta = torch.bmm(c.permute(2, 1, 0), w_grad.permute(1, 0, 2))
        if hparams.autograd_local_w and not hparams.autograd_local:
            raise Exception
        if not hparams.autograd_local_w:
            with torch.no_grad():
                params["W"][l_idx] = params["W"][l_idx] - lr(hparams, t) * w_delta
        if hparams.autograd_local:
            loss = L1_LOSS_FN(torch.sigmoid(h_out), targets)
            params["opt_l"][l_idx].zero_grad()
            loss.backward()
            if hparams.autograd_local_w:
                params["W"][l_idx].grad = w_delta
            params["opt_l"][l_idx].step()
        else:
            with torch.no_grad():
                # weights: [batch_size, layer_dim, input_dim]
                weights = torch.bmm(c.permute(2, 0, 1), params["W"][l_idx]).permute(
                    1, 0, 2
                )
                # Get new layer output with updated weights
                # h_out_updated: [batch_size, layer_dim]
                h_out_updated = torch.bmm(weights, h_in.unsqueeze(2)).squeeze(2)
    if not hparams.autograd_global:
        return h_out.detach(), params, h_out_updated
    else:
        return h_out, params, h_out_updated


def base_layer(s_bias, hparams):
    return s_bias[:, :-1]


def base_layer2(s_bias, hparams):
    return torch.clamp(
        torch.sigmoid(s_bias[:, :-1]),
        min=hparams["pred_clip"],
        max=1 - hparams["pred_clip"],
    ).detach()


def init_params(layer_sizes, hparams, binary_class=0):
    ctx, W, opt = [], [], []
    for i in range(1, len(layer_sizes)):
        input_dim, layer_dim = layer_sizes[i - 1], layer_sizes[i]
        layer_ctx = rand_hspace_dgn.get_params(hparams, layer_dim)
        layer_W = 0.5 * torch.ones(layer_dim, hparams["num_branches"], input_dim + 1)
        if hparams["gpu"]:
            layer_W = layer_W.cuda()
        ctx_param = torch.nn.Parameter(
            layer_ctx, requires_grad=(hparams.autograd_local or hparams.autograd_global)
        )
        W_param = torch.nn.Parameter(
            layer_W,
            requires_grad=(hparams.autograd_local_w or hparams.autograd_global_w),
        )

        if hparams.autograd_local and hparams.autograd_local_w:
            layer_opt = torch.optim.Adam(
                params=[W_param, ctx_param], lr=hparams.lr_autograd
            )
        elif hparams.autograd_local:
            layer_opt = torch.optim.Adam(params=[ctx_param], lr=hparams.lr_autograd)
        else:
            layer_opt = None

        ctx.append(ctx_param)
        W.append(W_param)
        opt.append(layer_opt)

    if hparams.autograd_global and hparams.autograd_global_w:
        opt_g = torch.optim.Adam(params=W + ctx, lr=hparams.lr_autograd)
    elif hparams.autograd_global:
        opt_g = torch.optim.Adam(params=ctx, lr=hparams.lr_autograd)
    else:
        opt_g = None

    return {"ctx": ctx, "W": W, "opt_l": opt, "opt_g": opt_g}
