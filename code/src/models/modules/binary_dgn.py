import torch
from math import e
from pytorch_lightning import LightningModule

from src.models.modules.rand_halfspace_dgn import RandHalfSpaceDGN
from src.utils.gated_plotter import GatedPlotter


class BinaryDGN(LightningModule):
    def __init__(self, hparams: dict, binary_class: int, X_all=None, y_all=None):
        super().__init__()
        # with torch.no_grad():
        self.automatic_optimization = False
        self.hparams = hparams
        self.ctx_params = []
        self.W = []
        self.t = 0
        self.ctx_params_bias = True
        self.s_dim = hparams["input_size"]
        self.num_neurons = (
            self.s_dim, hparams["lin1_size"], hparams["lin2_size"], hparams["lin3_size"])
        self.pred_clip = hparams["pred_clipping"]
        self.num_branches = hparams["num_branches"]
        self.num_layers_used = hparams["num_layers_used"]
        self.binary_class = binary_class
        # self.X_all = torch.cat(
        #     [X_all, torch.ones(X_all.shape[0], 1, device=self.device)], dim=1)

        self.init_params()

        def add_ctx_to_plot(xy, add_to_plot_fn):
            for l_idx in range(self.num_layers_used):
                Z = RandHalfSpaceDGN.calc_raw(xy, self.ctx_params[l_idx])
                for b_idx in range(self.num_branches):
                    add_to_plot_fn(Z[:, b_idx])

        if self.hparams["plot"]:
            self.plotter = GatedPlotter(X_all, y_all, add_ctx_to_plot)

    def init_params(self):
        # Context functions and weights for gated layers
        for i in range(1, len(self.num_neurons)):
            input_dim, layer_dim = self.num_neurons[i-1], self.num_neurons[i]
            layer_ctx_params = RandHalfSpaceDGN.get_params(self.s_dim + 1, layer_dim,
                                                           self.num_branches,
                                                           ctx_bias=self.ctx_params_bias,
                                                           pretrained_ctx=self.hparams["pretrained_ctx"])
            # if hparams["svm_context"] and i == 3:
            #     svm1_coef = torch.tensor(
            #         torch.load('../../../../svm1_coef.pt'))
            #     svm1_intercept = torch.tensor(
            #         torch.load('../../../../svm1_intercept.pt'))
            #     layer_ctx_params.hyperplanes[0, 0, :-1] = svm1_coef[binary_class]
            #     layer_ctx_params.hyperplanes[0, 0, -1] = svm1_intercept[binary_class]

            # layer_W = torch.full((layer_dim, self.num_branches, input_dim+1),
            #                      1.0/input_dim)
            layer_W = 0.5 * \
                torch.ones(layer_dim, self.num_branches, input_dim + 1)
            if self.hparams["gpu"]:
                self.ctx_params.append(layer_ctx_params.cuda())
                self.W.append(layer_W.cuda())
            else:
                self.ctx_params.append(layer_ctx_params)
                self.W.append(layer_W)

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
        # with torch.no_grad():
        layer_dim, _, input_dim = self.W[l_idx].shape
        layer_bias = e / (e+1)
        h = torch.cat(
            [h.detach(), layer_bias *
                torch.ones(h.shape[0], 1, device=self.device)],
            dim=1)
        assert(input_dim == h.shape[1])
        # c: [batch_size, layer_dim]
        c = RandHalfSpaceDGN.calc(
            s, self.ctx_params[l_idx], self.hparams["gpu"])
        # weights: [batch_size, layer_dim, input_dim]
        weights = torch.bmm(c.float().permute(2, 0, 1),
                            self.W[l_idx]).permute(1, 0, 2)
        # h_out: [batch_size, layer_dim]
        h_out = torch.bmm(weights, h.unsqueeze(2)).squeeze(2)
        # if is_train:
        #     t = y.unsqueeze(1)
        #     r_out = torch.sigmoid(h_out)
        #     r_out_clipped = torch.clamp(r_out,
        #                                 min=self.pred_clip,
        #                                 max=1-self.pred_clip)
        #     # learn_gates: [batch_size, layer_dim]
        #     learn_gates = (torch.abs(t - r_out) > self.pred_clip).float()
        #     w_grad1 = (r_out_clipped - t) * learn_gates
        #     w_grad2 = torch.bmm(w_grad1.unsqueeze(2), h.unsqueeze(1))
        #     w_delta = torch.bmm(c.float().permute(2, 1, 0),
        #                         w_grad2.permute(1, 0, 2))
        #     # assert(w_delta.shape == self.W[l_idx].shape)
        #     # TODO delete this: self.W.grad = w_delta
        #     # optimizer[].step(0)

        #     self.W[l_idx] -= self.lr() * w_delta
        return h_out

    def base_layer(self, s_bias):
        # with torch.no_grad():
        # h = torch.empty_like(s).copy_(s)
        # h = torch.clamp(torch.sigmoid(h), self.pred_clip, 1 - self.pred_clip)
        # h = torch.sigmoid(h)
        # return 0.5 * torch.ones_like(s_bias[:, :-1])
        return s_bias[:, :-1]

    def forward_helper(self, s_bias, y, is_train: bool):
        # with torch.no_grad():
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
        if len(s.shape) > 1:
            s = s.flatten(start_dim=1)
        s_bias = torch.cat(
            [s, torch.ones(s.shape[0], 1, device=self.device)], dim=1)
        h = self.forward_helper(s_bias, y, is_train)
        if is_train and self.hparams["plot"]:
            if self.binary_class == 0 and not (self.t % 5):
                #
                print('\ntemp\n', self.ctx_params[2])
                #

                def forward_fn(xy): return self.forward_helper(
                    xy, y=None, is_train=False)

                def add_ctx_to_plot(xy, add_to_plot_fn):
                    for l_idx in range(self.num_layers_used):
                        Z = RandHalfSpaceDGN.calc_raw(
                            xy, self.ctx_params[l_idx])
                        for b_idx in range(self.num_branches):
                            add_to_plot_fn(Z[:, b_idx])
                self.plotter.save_data(forward_fn, add_ctx_to_plot)
        return torch.sigmoid(h)

    def training_step(self, batch, batch_idx, optimizer_idx):
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        opt.step()
        return

    # With only neuron weight updating
    # def configure_optimizers(self):
    #     pass

    # With neuron weight updating and global straight-through estimators
    def configure_optimizers(self):
        return [torch.optim.Adam(params=i) for i in self.ctx_params]

    # With neuron weight updating and local straight-through estimators
    # def configure_optimizers(self):
    #     return [torch.optim.Adam(params=i) for i in self.ctx_params]
