import torch
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import LightningModule
from matplotlib.animation import FuncAnimation

from src.models.modules.rand_halfspace_dgn import RandHalfSpaceDGN


class BinaryDGN(LightningModule):
    def __init__(self, hparams: dict, binary_class: int, X_all=None, y_all=None):
        super().__init__()
        self.hparams = hparams
        self.ctx = []
        self.W = []
        self.t = 0
        self.ctx_bias = True
        self.s_dim = hparams["input_size"]
        self.num_neurons = (
            self.s_dim, hparams["lin1_size"], hparams["lin2_size"], hparams["lin3_size"])
        self.pred_clip = hparams["pred_clipping"]
        self.num_branches = hparams["num_branches"]
        self.num_layers_used = hparams["num_layers_used"]
        self.binary_class = binary_class
        self.X_all = X_all
        self.y_all = y_all
        # self.X_all = torch.cat(
        #     [X_all, torch.ones(X_all.shape[0], 1, device=self.device)], dim=1)
        self.init_params()
        self.init_plot()

    def init_params(self):
        # Context functions and weights for gated layers
        for i in range(1, len(self.num_neurons)):
            input_dim, layer_dim = self.num_neurons[i-1], self.num_neurons[i]
            layer_ctx = RandHalfSpaceDGN(self.s_dim + 1, layer_dim, self.num_branches,
                                         ctx_bias=self.ctx_bias,
                                         trained_ctx=self.hparams["trained_ctx"])
            # if hparams["svm_context"] and i == 3:
            #     svm1_coef = torch.tensor(
            #         torch.load('../../../../svm1_coef.pt'))
            #     svm1_intercept = torch.tensor(
            #         torch.load('../../../../svm1_intercept.pt'))
            #     layer_ctx.hyperplanes[0, 0, :-1] = svm1_coef[binary_class]
            #     layer_ctx.hyperplanes[0, 0, -1] = svm1_intercept[binary_class]

            # layer_W = torch.full((layer_dim, self.num_branches, input_dim+1),
            #                      1.0/input_dim)
            layer_W = 0.5 * \
                torch.zeros(layer_dim, self.num_branches, input_dim + 1)
            if self.hparams["gpu"]:
                self.ctx.append(layer_ctx.cuda())
                self.W.append(layer_W.cuda())
            else:
                self.ctx.append(layer_ctx)
                self.W.append(layer_W)

    def init_plot(self):
        # For plotting
        self.NX, self.NY = 120, 120
        self.Z_out_all = []
        self.plot, self.ax = plt.subplots()
        if self.hparams["plot"]:
            self.ax.scatter(self.X_all[:, 0], self.X_all[:, 1],
                            c=self.y_all, cmap='hot', marker='.',
                            linewidths=0.)
            self.ax = self.plot.gca()
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            xx = np.linspace(xlim[0], xlim[1], self.NX)  # X is from min to max
            # Y is from max to min to align
            yy = np.linspace(ylim[1], ylim[0], self.NY)
            XX, YY = np.meshgrid(xx, yy)
            self.xy = np.stack([XX.ravel(), YY.ravel(), np.ones(XX.size)]).T
            self.xy = torch.tensor(self.xy, dtype=torch.float)
            # output_shape = [self.num_branches] + list(XX.shape)
            for l_idx in range(self.num_layers_used):
                Z = self.ctx[l_idx].calc_raw(self.xy)
                for b_idx in range(self.num_branches):
                    Z_b = Z[:, b_idx].reshape(self.NX, self.NY)
                    self.ax.contour(XX, YY, Z_b, colors='k', levels=[0], alpha=0.5,
                                    linestyles=['-'])

    def lr(self):
        # return min(self.hparams["lr"], (1.1 * self.hparams["lr"])/(1.0 + 1e-2 * self.t))
        return self.hparams["lr"]

    def gated_layer(self, h, s, y, l_idx, is_train):
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
        layer_dim, _, input_dim = self.W[l_idx].shape
        h = torch.cat(
            [torch.ones(h.shape[0], 1, device=self.device), h], dim=1)
        assert(input_dim == h.shape[1])
        # [batch_size, layer_dim]
        c = self.ctx[l_idx].calc(s, self.hparams["gpu"])
        weights = torch.bmm(c.float().permute(2, 0, 1),
                            self.W[l_idx]).permute(1, 0, 2)  # [batch_size, layer_dim, input_dim]
        h_out = torch.bmm(weights,
                          h.unsqueeze(2)).squeeze(2)  # [batch_size, layer_dim]
        if is_train:
            t = y.unsqueeze(1)
            r_out = torch.sigmoid(h_out)
            r_out_clipped = torch.clamp(r_out,
                                        min=self.pred_clip,
                                        max=1-self.pred_clip)
            # r_out_clipped = r_out
            # [batch_size, layer_dim]
            learn_gates = (torch.abs(t - r_out) > self.pred_clip).float()
            w_grad1 = (r_out_clipped - t) * learn_gates
            w_grad2 = torch.bmm(w_grad1.unsqueeze(2), h.unsqueeze(1))
            w_delta = torch.bmm(c.float().permute(2, 1, 0),
                                w_grad2.permute(1, 0, 2))
            # assert(w_delta.shape == self.W[l_idx].shape)
            self.W[l_idx] -= self.lr() * w_delta
        return h_out

    def base_layer(self, s_bias):
        # h = torch.empty_like(s).copy_(s)
        # h = torch.clamp(torch.sigmoid(h), self.pred_clip, 1 - self.pred_clip)
        # h = torch.sigmoid(h)
        return 0.5 * torch.ones_like(s_bias[:, :-1])
        # return s_bias[:, :-1]

    def record_plot_info(self):
        Z_out = self.forward_helper(self.xy, y=None, is_train=False)
        self.Z_out_all.append(Z_out.reshape(self.NX, self.NY))

    def update_plot(self, Z_out):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.imshow(Z_out.reshape(self.NX, self.NY),
                       vmin=Z_out.min(), vmax=Z_out.max(), cmap='viridis',
                       extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                       interpolation='none')

    def save_animation(self, filename):
        if self.hparams["plot"] and self.binary_class == 0:
            anim = FuncAnimation(self.plot, self.update_plot,
                                 frames=self.Z_out_all)
            anim.save(filename, dpi=80, writer='imagemagick')

    def forward_helper(self, s_bias, y, is_train: bool):
        with torch.no_grad():
            h = self.base_layer(s_bias)
            for l_idx in range(self.num_layers_used):
                h = self.gated_layer(h, s_bias, y, l_idx, is_train)
        return h

    def forward(self, s, y, is_train: bool):
        """Calculate output of DGN for input x and side info s

        Args:
            s ([Float * [batch_size, self.s_dim]]): Batch of side info samples s
            y ([Float * [batch_size]]): Batch of binary targets
            is_train (bool): Whether to train/update on this batch or not
        Returns:
            [Float * [batch_size]]: Batch of DGN outputs (0 < probability < 1)
        """
        if is_train:
            self.t += 1
        s = s.flatten(start_dim=1)
        s_bias = torch.cat(
            [s, torch.ones(s.shape[0], 1, device=self.device)], dim=1)
        h = self.forward_helper(s_bias, y, is_train)
        if is_train and self.hparams["plot"] and not (self.t % 5) and self.binary_class == 0:
            self.record_plot_info()
        return torch.sigmoid(h)
