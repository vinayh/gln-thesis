import torch
from torch import nn
from pytorch_lightning import LightningModule


def logit(x):
    return torch.log(x / (torch.ones_like(x) - x))

def logit_geo_mix(logit_prev_layer, weights):
    """Geometric weighted mixture of prev_layer using weights

    Args:
        logit_prev_layer ([Float] * n_neurons): Logit of prev layer activations
        weights ([Float] * n_neurons): Weights for neurons in prev layer

    Returns:
        [Float]: sigmoid(w * logit(prev_layer)), i.e. output of current neuron
    """
    return torch.sigmoid(weights.matmul(logit_prev_layer))

class HalfSpace(LightningModule):
    def __init__(self, s_dim, layer_size, num_branches, ctx_bias=True):
        """Initialize half-space context layer of specified size and num contexts

        Args:
            s_dim (Int): Size of side info s
            layer_size (Int): Size of layer to which contexts are applied (output dim)
            num_branches (Int): Number of half-space gating branches
        """
        super().__init__()
        self.ctx_bias = ctx_bias
        self.n_branches = num_branches
        self.layer_size = layer_size
        self.register_buffer("bitwise_map", torch.tensor([2**i for i in range(self.n_branches)]))
        self.register_buffer("hyperplanes", torch.empty(layer_size, num_branches, s_dim + 1).normal_(mean=0, std=1.0))
        self.hyperplanes = self.hyperplanes / torch.linalg.norm(self.hyperplanes[:, :, :-1], axis=(1, 2))[:, None, None]

    def calc(self, s, gpu=False):
        """Calculates context indices for half-space gating given side info s

        Args:
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
            gpu (bool): Indicates whether model is running on GPU

        Returns:
            [Int * [batch_size, layer_size]]: Context indices for each side info
                                              sample in batch
        """
        return (torch.einsum('abc,dc->dba', self.hyperplanes, s) > 0).bool()
        # return ctx_results

class BinaryDGN(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        self.ctx = []
        self.W = []
        self.t = 0
        self.ctx_bias = True
        s_dim = hparams["input_size"]
        self.num_neurons = (s_dim, hparams["lin1_size"], hparams["lin2_size"], hparams["lin3_size"])
        # self.w_clip = hparams["weight_clipping"]
        self.pred_clip = hparams["pred_clipping"]
        num_branches = hparams["num_branches"]

        # Context functions and weights for gated layers
        for i in range(1, len(self.num_neurons)):  # Add ctx and w for each layer until the single output neuron layer
            input_dim, layer_dim = self.num_neurons[i-1], self.num_neurons[i]
            layer_ctx = HalfSpace(s_dim, layer_dim, num_branches, ctx_bias=self.ctx_bias)
            # layer_W = torch.full((layer_dim, num_branches, input_dim),
            #                      1.0/input_dim)
            layer_W = torch.zeros(layer_dim, num_branches, input_dim + 1)
            if self.hparams["gpu"]:
                self.ctx.append(layer_ctx.cuda())
                self.W.append(layer_W.cuda())
            else:
                self.ctx.append(layer_ctx)
                self.W.append(layer_W)
            continue
        return
    
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
        layer_dim = self.W[l_idx].shape[0]
        h = torch.cat([h, torch.ones(h.shape[0], 1, device=self.device)], dim=1)
        input_dim = h.shape[1]
        assert(input_dim == self.W[l_idx].shape[2])
        t = y.unsqueeze(1)
        c = self.ctx[l_idx].calc(s, self.hparams["gpu"])  # [batch_size, layer_dim]
        weights = torch.einsum('abc,cbd->acd', c.float(), self.W[l_idx])  # [batch_size, layer_dim, input_dim]
        h_out = torch.bmm(weights, h.unsqueeze(2)).squeeze(2)  # [batch_size, layer_dim]
        if is_train:
            r_out = torch.sigmoid(h_out)
            r_out_clipped = torch.clamp(r_out, self.pred_clip, 1-self.pred_clip)
            learn_gates = (torch.abs(t - r_out) > self.pred_clip).float()  # [batch_size, layer_dim]
            w_grad1 = (r_out_clipped - t) * learn_gates
            w_grad2 = torch.bmm(w_grad1.unsqueeze(2), h.unsqueeze(1))
            w_delta = torch.bmm(c.float().permute(2,1,0), w_grad2.permute(1,0,2))
            assert(w_delta.shape == self.W[l_idx].shape)
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
        # self.lr = min(0.6, 1.0/(1.0 + 1e-2 * self.t))
        self.lr = 1e-4
        s = s.flatten(start_dim=1)
        h = torch.empty_like(s).copy_(s)
        s = torch.cat([s, torch.ones(batch_size, 1, device=self.device)], dim=1)
        h = torch.clamp(torch.sigmoid(h), self.pred_clip, 1 - self.pred_clip)
        with torch.no_grad():
            h = self.gated_layer(h, s, y, 0, is_train)
            h = self.gated_layer(h, s, y, 1, is_train)
            h = self.gated_layer(h, s, y, 2, is_train)
        return torch.sigmoid(h)
