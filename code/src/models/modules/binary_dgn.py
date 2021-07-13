import torch
from math import e

from src.models.modules.rand_halfspace_dgn import RandHalfSpaceDGN
# from src.utils.gated_plotter import GatedPlotter


def init_params(num_neurons, hparams, binary_class=0, X_all=None, y_all=None):
    ctx, W, opt = [], []
    for i in range(1, len(num_neurons)):
        with torch.no_grad():
            input_dim, layer_dim = num_neurons[i-1], num_neurons[i]
            layer_ctx = RandHalfSpaceDGN.get_params(hparams["input_size"] + 1, layer_dim,
                                                    hparams["num_branches"],
                                                    ctx_bias=True,
                                                    pretrained_ctx=hparams["pretrained_ctx"])
            layer_W = 0.5 * \
                torch.ones(layer_dim, hparams["num_branches"], input_dim + 1)
            layer_opt = torch.optim.Adam(params=[layer_ctx])
        if hparams["gpu"]:
            ctx.append(torch.nn.Parameter(
                layer_ctx, requires_grad=True).cuda())
            W.append(torch.nn.Parameter(layer_W, requires_grad=False).cuda())
            opt.append(layer_opt.cuda())
        else:
            ctx.append(torch.nn.Parameter(layer_ctx, requires_grad=True))
            W.append(torch.nn.Parameter(layer_W, requires_grad=False))
            opt.append(layer_opt)
    return {"ctx": ctx, "weights": W, "opt": opt}


def lr(hparams):
    # return min(hparams["lr"], (1.1 * hparams["lr"])/(1.0 + 1e-2 * t))
    return hparams["lr"]


def gated_layer(params, hparams, h, s, y, l_idx, is_train, is_gpu):
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
    input_dim = hparams["input_size"] + 1
    layer_bias = e / (e+1)
    h = torch.cat([h, layer_bias * torch.ones(h.shape[0], 1)], dim=1)
    assert(input_dim == h.shape[1])
    # c: [batch_size, layer_dim]
    c = RandHalfSpaceDGN.calc(
        s, params["ctx"][l_idx], is_gpu)
    # weights: [batch_size, layer_dim, input_dim]
    weights = torch.bmm(c.float().permute(2, 0, 1),
                        params["weights"][l_idx]).permute(1, 0, 2)
    # h_out: [batch_size, layer_dim]
    h_out = torch.bmm(weights, h.unsqueeze(2)).squeeze(2)
    if is_train:
        t = y.unsqueeze(1)
        r_out = torch.sigmoid(h_out)
        r_out_clipped = torch.clamp(r_out,
                                    min=hparams["pred_clipping"],
                                    max=1-hparams["pred_clipping"])
        # learn_gates: [batch_size, layer_dim]
        learn_gates = (torch.abs(t - r_out) > hparams["pred_clipping"]).float()
        w_grad1 = (r_out_clipped - t) * learn_gates
        w_grad2 = torch.bmm(w_grad1.unsqueeze(2), h.unsqueeze(1))
        w_delta = torch.bmm(c.float().permute(2, 1, 0),
                            w_grad2.permute(1, 0, 2))
        # assert(w_delta.shape == W[l_idx].shape)
        # TODO delete this: W.grad = w_delta
        # optimizer[].step(0)

        params["weights"][l_idx] -= lr(hparams) * w_delta
    return h_out, params


def base_layer(s_bias):
    # h = torch.empty_like(s).copy_(s)
    # h = torch.clamp(torch.sigmoid(h), hparams["pred_clipping"], 1 - hparams["pred_clipping"])
    # h = torch.sigmoid(h)
    # return 0.5 * torch.ones_like(s_bias[:, :-1])
    return s_bias[:, :-1]


def forward(params, binary_class, hparams, t, s, y, is_train: bool):
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
    s_bias = torch.cat(
        [s, torch.ones(s.shape[0], 1, device=hparams.device)], dim=1)
    h = base_layer(s_bias)
    for l_idx in range(hparams["num_layers_used"]):
        h, _ = gated_layer(params, hparams, h, s_bias, y,
                           l_idx, is_train, is_gpu=False)
    return torch.sigmoid(h)


# def configure_optimizers(params):  # Global straight-through, NO neuron updates
#     return [torch.optim.SGD(params=params["ctx"], lr=0.1)]
#     # return [torch.optim.Adam(params=params["ctx"], lr=0.1)]

# def configure_optimizers(params):  # Only neuron weight updating
#     pass


# def configure_optimizers(params):  # Neuron updates and global straight-through
#     pass

# def configure_optimizers(params):  # Neuron updates, local straight-through
#     return [torch.optim.Adam(params=i) for i in params["ctx"]]

def configure_optimizers(params):  # Local straight-through, NO neuron updates
    return [torch.optim.Adam(params=[ctx_layer]) for ctx_layer in params["ctx"]]


######
# Old
######

# class BinaryDGN(LightningModule):
    # def add_ctx_to_plot(xy, add_to_plot_fn):
    #     for l_idx in range(hparams["num_layers_used"]):
    #         Z = RandHalfSpaceDGN.calc_raw(xy, params["ctx"][l_idx])
    #         for b_idx in range(self.num_branches):
    #             add_to_plot_fn(Z[:, b_idx])

    # if hparams["plot"]:
    #     self.plotter = GatedPlotter(X_all, y_all, add_ctx_to_plot)

    # def training_step(self, batch, batch_idx, optimizer_idx):
    #     opt = self.optimizers()
    #     opt.zero_grad()
    #     loss = self.compute_loss(batch)
    #     self.manual_backward(loss)
    #     opt.step()
    #     return
