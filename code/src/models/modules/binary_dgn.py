import torch
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import LightningModule

from src.models.modules.rand_halfspace_dgn import RandHalfSpaceDGN


class BinaryDGN(LightningModule):
    def __init__(self, hparams: dict, binary_class: int, X_all=None, y_all=None):
        super().__init__()
        self.hparams = hparams
        self.ctx = []
        self.W = []
        self.t = 0
        self.ctx_bias = True
        s_dim = hparams["input_size"]
        self.num_neurons = (
            s_dim, hparams["lin1_size"], hparams["lin2_size"], hparams["lin3_size"])
        self.pred_clip = hparams["pred_clipping"]
        self.num_branches = hparams["num_branches"]
        self.num_layers_used = hparams["num_layers_used"]
        self.binary_class = binary_class
        # self.X_all = torch.cat(
        #     [X_all, torch.ones(X_all.shape[0], 1, device=self.device)], dim=1)
        self.X_all = X_all
        self.y_all = y_all

        # Context functions and weights for gated layers
        for i in range(1, len(self.num_neurons)):
            input_dim, layer_dim = self.num_neurons[i-1], self.num_neurons[i]
            layer_ctx = RandHalfSpaceDGN(s_dim + 1, layer_dim, self.num_branches,
                                         ctx_bias=self.ctx_bias,
                                         trained_ctx=hparams["trained_ctx"])

            # if hparams["svm_context"] and i == 3:
            #     svm1_coef = torch.tensor(
            #         torch.load('../../../../svm1_coef.pt'))
            #     svm1_intercept = torch.tensor(
            #         torch.load('../../../../svm1_intercept.pt'))
            #     layer_ctx.hyperplanes[0, 0, :-1] = svm1_coef[binary_class]
            #     layer_ctx.hyperplanes[0, 0, -1] = svm1_intercept[binary_class]

            layer_W = torch.full((layer_dim, self.num_branches, input_dim+1),
                                 1.0/input_dim)
            # layer_W = 0.5 * \
            #     torch.ones(layer_dim, self.num_branches, input_dim + 1)
            if self.hparams["gpu"]:
                self.ctx.append(layer_ctx.cuda())
                self.W.append(layer_W.cuda())
            else:
                self.ctx.append(layer_ctx)
                self.W.append(layer_W)

    def lr(self):
        # return min(self.hparams["lr"], (1.1 * self.hparams["lr"])/(1.0 + 1e-2 * self.t))
        return self.hparams["lr"]

    def gated_layer(self, h, s, y, l_idx, is_train):
        """Using provided input activations, context functions, and weights,
           returns the result of the DGN layer

        Args:
            h ([Float * [batch_size, input_dim]]): Input activations with logit
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
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
        # return 0.5 * torch.ones_like(s_bias[:, :-1])
        return s_bias[:, :-1]

    def plot(self):
        """For 2D data (e.g. toy clustering dataset)
        """
        plt.scatter(self.X_all[:, 0], self.X_all[:, 1],
                    c=self.y_all, cmap='Reds')
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        NX, NY = 120, 120
        xx = np.linspace(xlim[0], xlim[1], NX)  # X is from min to max
        yy = np.linspace(ylim[1], ylim[0], NY)  # Y is from max to min to align
        XX, YY = np.meshgrid(xx, yy)
        xy = np.stack([XX.ravel(), YY.ravel(), np.ones(XX.size)]).T
        xy = torch.tensor(xy, dtype=torch.float)
        # output_shape = [self.num_branches] + list(XX.shape)
        for l_idx in range(self.num_layers_used):
            Z = self.ctx[l_idx].calc_raw(xy)
            for b_idx in range(self.num_branches):
                Z_b = Z[:, b_idx].reshape(NX, NY)
                ax.contour(XX, YY, Z_b, colors='k', levels=[0], alpha=0.5,
                           linestyles=['-'])
        if self.hparams["plot_decisions"]:
            # Should probably use plt.imshow here instead of plt.contourf
            # for proper continuous color gradients
            Z_out = self.forward_helper(xy, y=None, is_train=False)
            # min = Z_out.min()
            # max = Z_out.max()+0.001
            # clev = np.linspace(min, max, 100)
            # ax.contourf(XX, YY, Z_out.reshape(NX, NY), clev, alpha=0.2)
            ax.imshow(Z_out.reshape(NX, NY),
                      vmin=Z_out.min(), vmax=Z_out.max(), cmap='viridis',
                      extent=[XX.min(), XX.max(), YY.min(), YY.max()],
                      interpolation='none')
        plt.show()

    def forward_helper(self, s_bias, y, is_train: bool):
        with torch.no_grad():
            h = self.base_layer(s_bias)
            for l_idx in range(self.num_layers_used):
                h = self.gated_layer(h, s_bias, y, l_idx, is_train)
        return h

    def forward(self, s, y, is_train: bool):
        """Calculate output of DGN for input x and side info s

        Args:
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
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
        if self.hparams["plot"] and not (self.t % 20) and self.binary_class == 1:
            self.plot()
        return torch.sigmoid(h)
