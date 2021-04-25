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
    # return torch.sigmoid(torch.sum(weights * logit_prev_layer, dim=-1))
    return torch.sigmoid(weights.matmul(logit_prev_layer))

class HalfSpace(LightningModule):
    def __init__(self, s_dim, layer_size, num_subcontexts, ctx_bias=True):
        """Initialize half-space context layer of specified size and num contexts

        Args:
            s_dim (Int): Size of side info s
            layer_size (Int): Size of layer to which contexts are applied (output dim)
            num_subcontexts (Int): Number of half-space subcontexts
                                   i.e. num_contexts = 2**num_subcontexts
        """
        super().__init__()
        self.ctx_bias = ctx_bias
        self.n_subctx = num_subcontexts
        self.layer_size = layer_size
        self.register_buffer("bitwise_map", torch.tensor([2**i for i in range(self.n_subctx)]))
        if self.ctx_bias:
            input_dim = s_dim + 1
        else:
            input_dim = s_dim
        self.register_buffer("ctx_weights", torch.empty(num_subcontexts, input_dim, layer_size).normal_(mean=0.5, std=1.0))
        # Init subcontext functions (half-space gatings)
        # self.subctx_fn = []
        # for _ in range(self.n_subctx):
        #     new_subctx = nn.Linear(s_dim, layer_size)
        #     nn.init.normal_(new_subctx.weight, mean=0.0, std=0.1)
        #     for param in new_subctx.parameters():
        #         param.requires_grad = False
        #     self.subctx_fn.append(new_subctx)

    def calc(self, s, gpu=False):
        """Calculates context indices for half-space gating given side info s

        Args:
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s

        Returns:
            [Int * [batch_size, layer_size]]: Context indices for each side info
                                              sample in batch
        """
        # def get_subctx_val(idx):
        #     return (self.subctx_fn[idx](s.flatten(start_dim=2)) > 0).squeeze(dim=1) * self.bitwise_map[idx]
        # ctx = torch.sum(torch.stack([get_subctx_val(i) for i in range(self.n_subctx)]), dim=0)
        # return ctx
        ctx_results = (torch.einsum('abc,db->dca', self.ctx_weights, s) > 0)
        if gpu:
            contexts = ctx_results.float().matmul(self.bitwise_map.float()).long()
        else:
            contexts = ctx_results.long().matmul(self.bitwise_map)
        return contexts

class BinaryGLN(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        self.ctx = []
        self.W = []
        self.lr = 0.4
        self.t = 0
        self.l_sizes = (hparams["lin1_size"], hparams["lin2_size"], hparams["lin3_size"], 1)
        self.ctx_bias = True
        # Weights for base predictor (arbitrary activations)
        # self.W_base = torch.rand((self.l_sizes[0], self.l_sizes[0]), 1.0/self.l_sizes[0])
        self.W_base = nn.Linear(self.l_sizes[0], self.l_sizes[0])
        torch.nn.init.normal_(self.W_base.weight.data, mean=0.0, std=0.2)
        torch.nn.init.normal_(self.W_base.bias.data, mean=0.0, std=0.2)
        self.W_base.requires_grad_ = False
        # Context functions and weights for gated layers
        s_dim = hparams["input_size"]
        for i in range(len(self.l_sizes)-1):  # Add ctx and w for each layer until the single output neuron layer
            curr_layer_dim, next_layer_dim = self.l_sizes[i], self.l_sizes[i+1]
            if self.hparams["gpu"]:
                self.ctx.append(HalfSpace(s_dim,
                                          curr_layer_dim,
                                          hparams["num_subcontexts"],
                                          ctx_bias=self.ctx_bias).cuda())
                self.W.append(torch.full((curr_layer_dim,
                                      2**hparams["num_subcontexts"],
                                      next_layer_dim),
                                     1.0/curr_layer_dim).cuda())
            else:
                self.ctx.append(HalfSpace(s_dim,
                                        curr_layer_dim,
                                        hparams["num_subcontexts"],
                                        ctx_bias=self.ctx_bias))
                self.W.append(torch.full((curr_layer_dim,
                                        2**hparams["num_subcontexts"],
                                        next_layer_dim),
                                        1.0/curr_layer_dim))
    
    def gated_layer(self, logit_x, s, y, l_idx, train):
        """Using provided input activations, context functions, and weights,
           returns the result of the GLN layer

        Args:
            logit_x ([Float * [batch_size, input_layer_dim]]): Input activations (with logit applied)
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
            y ([Float * [batch_size]]): Batch of binary targets
            l_idx (Int): Index of layer to use for selecting correct ctx and layer weights
            train (bool): Whether to train/update on this batch or not (for val/test)
        Returns:
            [Float * [batch_size, output_layer_dim]]: Output of GLN layer
        """
        batch_size = s.shape[0]
        input_layer_dim = self.W[l_idx].shape[0]
        w_clip = self.hparams["weight_clipping"]
        # assert(self.W[l_idx].shape[0] == logit_x.shape[1])
        # assert(logit_x.shape[0] == batch_size)
        c = self.ctx[l_idx].calc(s, self.hparams["gpu"])  # [batch_size, input_layer_dim]
        # assert(c.shape[0] == batch_size)
        # assert(len(c[0]) == input_layer_dim)
        w_ctx = torch.stack([self.W[l_idx][range(input_layer_dim), c[j], :] for j in range(batch_size)])  # [batch_size, input_layer_dim, output_layer_dim]
        logit_preds = torch.bmm(w_ctx.permute(0,2,1), logit_x.unsqueeze(2)).flatten(start_dim=1)  # [batch_size, output_layer_dim]
        if train:
            loss = torch.sigmoid(logit_preds) - y.unsqueeze(1)  # [batch_size, output_layer_dim]
            w_delta = torch.einsum('ab,ac->acb', loss, logit_x)  # [batch_size, input_layer_dim, output_layer_dim]
            w_new = torch.clamp(w_ctx - self.lr * w_delta, min=-w_clip, max=w_clip)  # [batch_size, input_layer_dim, output_layer_dim]
            for j in range(batch_size):
                self.W[l_idx][range(input_layer_dim), c[j], :] = w_new[j]
        return logit_preds

    def base_layer_old(self, s, y, layer_size):
        batch_size = s.shape[0]
        clip_param = self.hparams["pred_clipping"]
        rand_activations = torch.empty(batch_size, self.l_sizes[0], device=self.device).normal_(mean=0.5, std=1.0)
        # rand_activations.requires_grad = False
        # x = self.W_base(rand_activations)
        # x = torch.clamp(x, min=clip_param, max=1.0-clip_param)
        x = torch.clamp(rand_activations, min=clip_param, max=1.0-clip_param)
        x = logit(x)
        return torch.ones(batch_size, layer_size, device=self.device)/2.0
    
    def base_layer(self, s, y, layer_size):
        mean = torch.mean(s, dim=0)
        stdev = torch.std(s, dim=0)
        x = (s - mean) / (stdev + 1.0)
        return x

    def forward(self, s, y, train: bool):
        """Calculate output of Gated Linear Network for input x and side info s

        Args:
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
            y ([Float * [batch_size]]): Batch of binary targets
            train (bool): Whether to train/update on this batch or not (for val/test)
        Returns:
            [Float * [batch_size]]: Batch of GLN outputs (0 < probability < 1)
        """
        batch_size = s.shape[0]
        s = s.flatten(start_dim=1)
        if self.ctx_bias:
            s = torch.cat([s, torch.ones(batch_size, 1, device=self.device)], dim=1)
        if train:
            self.t += 1
        self.lr = min(0.6, 1.0/(1.0 + 1e-2 * self.t))
        with torch.no_grad():
            # Base predictor followed by gated layers
            x = self.base_layer(s, y, self.l_sizes[0])
            x = self.gated_layer(x, s, y, 0, train)
            x = self.gated_layer(x, s, y, 1, train)
            x = self.gated_layer(x, s, y, 2, train)
        return torch.sigmoid(x)
