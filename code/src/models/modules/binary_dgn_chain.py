import torch

from pytorch_lightning import LightningModule
from src.models.modules.rand_halfspace_dgn import RandHalfSpaceDGN


class BinaryDGNChain(LightningModule):
    def __init__(self, hparams: dict, binary_class: int, pretrained: torch.Tensor):
        super().__init__()
        self.hparams = hparams
        self.ctx = []
        self.W = []
        self.t = 0
        self.pretrained = pretrained
        self.ctx_bias = True
        s_dim = hparams["input_size"]
        self.num_neurons = (s_dim, hparams["lin1_size"], hparams["lin2_size"], hparams["lin3_size"])
        self.pred_clip = hparams["pred_clipping"]
        num_branches = hparams["num_branches"]

        # Context functions and weights for gated layers
        for i in range(1, len(self.num_neurons)):  # For each layer
            input_dim, layer_dim = self.num_neurons[i-1], self.num_neurons[i]
            layer_ctx = RandHalfSpaceDGN(s_dim + 1, layer_dim, num_branches,
                                         ctx_bias=self.ctx_bias,
                                         trained_ctx=hparams["trained_ctx"])
            # Set weights of first neuron/branch in layer to that layer's pretrained weights/coef
            assert(layer_ctx.hyperplanes[0,0,:].shape == pretrained[i-1, :].shape)
            layer_ctx.hyperplanes[0, :, :] = pretrained[i-1, :]

            layer_W = torch.zeros(layer_dim, num_branches, input_dim + 1)
            if self.hparams["gpu"]:
                self.ctx.append(layer_ctx.cuda())
                self.W.append(layer_W.cuda())
            else:
                self.ctx.append(layer_ctx)
                self.W.append(layer_W)
    
    def gated_layer(self, h, s, y, l_idx, is_train):
        """Using provided input activations, context functions, and weights,
           returns the result of the DGN layer

        Args:
            h ([Float * [batch_size, input_dim]]): Input activations (with logit applied)
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
            y ([Float * [batch_size]]): Batch of binary targets
            l_idx (Int): Index of layer to use for selecting correct ctx and layer weights
            is_train (bool): Whether to train/update on this batch or not (for val/test)
        Returns:
            [Float * [batch_size, layer_dim]]: Output of DGN layer
        """
        batch_size = s.shape[0]
        layer_dim, _, input_dim = self.W[l_idx].shape
        h = torch.cat([torch.ones(h.shape[0], 1, device=self.device), h], dim=1)
        assert(input_dim == h.shape[1])
        t = y.unsqueeze(1)
        c = self.ctx[l_idx].calc(s, self.hparams["gpu"])  # [batch_size, layer_dim]
        weights = torch.bmm(c.float().permute(2,0,1), self.W[l_idx]).permute(1,0,2)  # [batch_size, layer_dim, input_dim]
        h_out = torch.bmm(weights, h.unsqueeze(2)).squeeze(2)  # [batch_size, layer_dim]
        if is_train:
            r_out = torch.sigmoid(h_out)
            # r_out_clipped = torch.clamp(r_out, self.pred_clip, 1-self.pred_clip)
            r_out_clipped = r_out
            learn_gates = (torch.abs(t - r_out) > self.pred_clip).float()  # [batch_size, layer_dim]
            w_grad1 = (r_out_clipped - t) * learn_gates
            w_grad2 = torch.bmm(w_grad1.unsqueeze(2), h.unsqueeze(1))
            w_delta = torch.bmm(c.float().permute(2,1,0), w_grad2.permute(1,0,2))
            # assert(w_delta.shape == self.W[l_idx].shape)
            self.W[l_idx] -= self.lr * w_delta
        return h_out

    def forward(self, s, y, is_train: bool):
        """Calculate output of DGN for input x and side info s

        Args:
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
            y ([Float * [batch_size]]): Batch of binary targets
            is_train (bool): Whether to train/update on this batch or not (for val/test)
        Returns:
            [Float * [batch_size]]: Batch of DGN outputs (0 < probability < 1)
        """
        batch_size = s.shape[0]
        if is_train:
            self.t += 1
        # self.lr = min(self.hparams["lr"], (1.1*self.hparams["lr"])/(1.0 + 1e-2 * self.t))
        self.lr = self.hparams["lr"]
        s = s.flatten(start_dim=1)
        h = torch.empty_like(s).copy_(s)
        s = torch.cat([s, torch.ones(batch_size, 1, device=self.device)], dim=1)
        # h = torch.clamp(torch.sigmoid(h), self.pred_clip, 1 - self.pred_clip)
        # h = torch.sigmoid(h)
        h = 0.5 * torch.ones_like(h)
        with torch.no_grad():
            h = self.gated_layer(h, s, y, 0, is_train)
            # h = self.gated_layer(h, s, y, 1, is_train)
            # h = self.gated_layer(h, s, y, 2, is_train)
        return torch.sigmoid(h)
