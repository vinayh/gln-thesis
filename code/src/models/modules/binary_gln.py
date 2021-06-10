import torch
from math import e
import matplotlib.pyplot as plt

from pytorch_lightning import LightningModule
from matplotlib.animation import FuncAnimation
import numpy as np

from src.models.modules.rand_halfspace_gln import RandHalfSpaceGLN
from src.utils.helpers import logit
from src.utils.gated_plotter import GatedPlotter


class BinaryGLN(LightningModule):
    def __init__(self, hparams: dict, binary_class: int, X_all=None, y_all=None):
        super().__init__()
        self.hparams = hparams
        self.ctx = []
        self.W = []
        self.t = 0
        s_dim = hparams["input_size"]
        self.l_sizes = (s_dim, hparams["lin1_size"],
                        hparams["lin2_size"], hparams["lin3_size"])
        self.ctx_bias = True
        self.w_clip = hparams["weight_clipping"]
        s_dim = hparams["input_size"]
        self.num_layers_used = hparams["num_layers_used"]
        self.num_subctx = hparams["num_subcontexts"]
        self.num_ctx = 2**self.num_subctx
        self.X_all = X_all
        self.y_all = y_all
        self.binary_class = binary_class
        # Weights for base predictor (arbitrary activations)
        # self.W_base = torch.nn.Linear(self.l_sizes[0], self.l_sizes[0])
        # self.W_base.requires_grad_ = False
        # torch.nn.init.normal_(self.W_base.weight.data, mean=0.0, std=0.2)
        # torch.nn.init.normal_(self.W_base.bias.data, mean=0.0, std=0.2)

        # Context functions and weights for gated layers
        # Add ctx and w for each layer until the single output neuron layer
        for i in range(1, len(self.l_sizes)):
            input_dim, layer_dim = self.l_sizes[i-1], self.l_sizes[i]
            new_ctx = RandHalfSpaceGLN(s_dim + 1, layer_dim, hparams["num_subcontexts"],
                                       ctx_bias=self.ctx_bias)
            new_W = torch.full((layer_dim, self.num_ctx,
                               input_dim + 1), 1.0/input_dim)

            if self.hparams["gpu"]:
                self.ctx.append(new_ctx.cuda())
                self.W.append(new_W.cuda())
            else:
                self.ctx.append(new_ctx)
                self.W.append(new_W)

        def add_ctx_to_plot(xy, add_to_plot_fn):
            for l_idx in range(self.num_layers_used):
                Z = self.ctx[l_idx].calc_raw(xy)[:, 0, :]
                for b_idx in range(self.num_subctx):
                    add_to_plot_fn(Z[:, b_idx])

        if self.hparams["plot"]:
            self.plotter = GatedPlotter(X_all, y_all, add_ctx_to_plot)

    def lr(self):
        # return 0.1
        return min(0.1, 0.2/(1.0 + 1e-2 * self.t))

    def gated_layer(self, logit_x, s, y, l_idx, is_train):
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
        layer_dim, _, input_dim = self.W[l_idx].shape
        # c: [batch_size, input_dim]
        c = self.ctx[l_idx].calc(s, self.hparams["gpu"])
        layer_bias = e / (e+1)
        logit_x = torch.cat([logit_x,
                             layer_bias * torch.ones(
                                 logit_x.shape[0], 1, device=self.device)],
                            dim=1)
        # w_ctx: [batch_size, input_dim, output_layer_dim]
        w_ctx = torch.stack([self.W[l_idx][range(layer_dim), c[j], :]
                             for j in range(batch_size)])
        logit_out = torch.bmm(w_ctx, logit_x.unsqueeze(2)).flatten(
            start_dim=1)  # [batch_size, output_layer_dim]
        if is_train:
            # loss: [batch_size, output_layer_dim]
            loss = torch.sigmoid(logit_out) - y.unsqueeze(1)
            # w_delta: torch.einsum('ab,ac->acb', loss, logit_x)  # [batch_size, input_dim, output_layer_dim]
            w_delta = torch.bmm(loss.unsqueeze(2), logit_x.unsqueeze(1))
            w_new = torch.clamp(w_ctx - self.lr() * w_delta,
                                min=-self.w_clip, max=self.w_clip)
            # [batch_size, input_dim, output_layer_dim]
            for j in range(batch_size):
                self.W[l_idx][range(layer_dim), c[j], :] = w_new[j]
        return logit_out

    def base_layer_old(self, s, y, layer_size):
        batch_size = s.shape[0]
        clip_param = self.hparams["pred_clipping"]
        rand_activations = torch.empty(
            batch_size, self.l_sizes[0], device=self.device).normal_(mean=0.5, std=1.0)
        # rand_activations.requires_grad = False
        # x = self.W_base(rand_activations)
        # x = torch.clamp(x, min=clip_param, max=1.0-clip_param)
        x = torch.clamp(rand_activations, min=clip_param, max=1.0-clip_param)
        x = logit(x)
        return torch.ones(batch_size, layer_size, device=self.device)/2.0

    def base_layer(self, s_bias, y, layer_size):
        # mean = torch.mean(s, dim=0)
        # stdev = torch.std(s, dim=0)
        # x = (s - mean) / (stdev + 1.0)
        return s_bias[:, :-1]

    def forward_helper(self, s_bias, y, is_train: bool):
        with torch.no_grad():
            x = self.base_layer(s_bias, y, self.l_sizes[0])
            # Gated layers
            x = self.gated_layer(x, s_bias, y, 0, is_train)
            # x = self.gated_layer(x, s_bias, y, 1, is_train)
            # x = self.gated_layer(x, s_bias, y, 2, is_train)
        return x

    def forward(self, s, y, is_train: bool):
        """Calculate output of Gated Linear Network for input x and side info s

        Args:
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
            y ([Float * [batch_size]]): Batch of binary targets
            is_train (bool): Whether to train/update on this batch or not (for val/test)
        Returns:
            [Float * [batch_size]]: Batch of GLN outputs (0 < probability < 1)
        """
        if is_train:
            self.t += 1
        s = s.flatten(start_dim=1)
        s_bias = torch.cat(
            [s, torch.ones(s.shape[0], 1, device=self.device)], dim=1)
        x = self.forward_helper(s_bias, y, is_train)
        if is_train and self.hparams["plot"]:
            if self.binary_class == 0 and not (self.t % 5):
                self.plotter.save_data(
                    lambda xy: self.forward_helper(xy, y=None, is_train=False))
        return torch.sigmoid(x)
