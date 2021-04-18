import torch
from torch import nn
from pytorch_lightning import LightningModule


def logit(x):
    return torch.log(x / (torch.ones_like(x) - x))

def logit_geo_mix(logit_prev_layer, weights):
    """Geometric weighted mixture of prev_layer using weights

    Args:
        weights ([Float] * n_neurons): Weights for neurons in prev layer
        prev_layer ([Float] * n_neurons): Activations of prev layer

    Returns:
        [Float]: sigmoid(w * logit(prev_layer)), i.e. output of current neuron
    """
    return torch.sigmoid(torch.sum(weights * logit_prev_layer, dim=-1))


class HalfSpace(LightningModule):
    def __init__(self, s_dim, layer_size, num_subcontexts):
        """Initialize half-space context layer of specified size and num contexts

        Args:
            s_dim (Int): Size of side info s
            layer_size (Int): Size of layer to which contexts are applied (output dim)
            num_subcontexts (Int): Number of half-space subcontexts
                                   i.e. num_contexts = 2**num_subcontexts
        """
        super().__init__()
        self.n_subctx = num_subcontexts
        self.bitwise_map = torch.tensor([2**i for i in range(self.n_subctx)], device=self.device)
        # Init subcontext functions (half-space gatings)
        self.subctx_fn = []
        for _ in range(self.n_subctx):
            # new_subctx = nn.Linear(s_dim, layer_size).cuda()
            new_subctx = nn.Linear(s_dim, layer_size)
            nn.init.normal_(new_subctx.weight, mean=0.0, std=0.1)
            self.subctx_fn.append(new_subctx)

    def calc(self, s):
        """Calculates context indices for half-space gating given side info s

        Args:
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s

        Returns:
            [Int * [batch_size, layer_size]]: Context indices for each side info
                                              sample in batch
        """
        batch_size = s.shape[0]
        layer_size = self.subctx_fn[0].out_features
        ctx = torch.zeros((batch_size, layer_size),
                          dtype=torch.long, device=self.device)
        s_view = s.view(s.shape[0], s.shape[1], -1)
        for i in range(self.n_subctx):
            ctx += (self.subctx_fn[i](s_view) > 0).squeeze() * self.bitwise_map[i]
        return ctx

class BinaryGLN(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        self.ctx = []
        self.W = []
        self.l_sizes = (hparams["lin1_size"], hparams["lin2_size"], hparams["lin3_size"], 1)
        # Weights for base predictor (arbitrary activations)
        # self.W_base = torch.rand((self.l_sizes[0], self.l_sizes[0]), 1.0/self.l_sizes[0])
        self.W_base = nn.Linear(self.l_sizes[0], self.l_sizes[0])
        self.W_base.requires_grad_ = False
        # Context functions and weights for gated layers
        s_dim = hparams["input_size"]
        for i in range(len(self.l_sizes)-1):  # Add ctx and w for each layer until the single output neuron layer
            curr_layer_dim, next_layer_dim = self.l_sizes[i], self.l_sizes[i+1]
            # self.ctx.append(HalfSpace(s_dim,
            #                           curr_layer_dim,
            #                           hparams["num_subcontexts"]).cuda())
            self.ctx.append(HalfSpace(s_dim,
                                      curr_layer_dim,
                                      hparams["num_subcontexts"]))
            self.W.append(torch.full((next_layer_dim,
                                      2**hparams["num_subcontexts"],
                                      curr_layer_dim),
                                     1.0/curr_layer_dim))
    
    def gated_layer(self, x, s, l_idx):
        """Using provided input activations, context functions, and weights,
           returns the result of the GLN layer

        Args:
            x ([Float * [batch_size, input_layer_dim]]): Input activations (with logit applied)
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
            ctx (HalfSpace): Instance of half-space context class
            w ([Float * [next_layer_dim, num_ctx, input_layer_dim]]): Layer weights
        
        Returns:
            [Float * [batch_size, next_layer_dim]]: Output of GLN layer
        """
        batch_size = s.shape[0]
        next_layer_dim, _, input_layer_dim = self.W[l_idx].shape
        weight_clip = self.hparams["weight_clipping"]
        assert(self.W[l_idx].shape[2] == x.shape[1])
        assert(x.shape[0] == batch_size)
        c = self.ctx[l_idx].calc(s)  # [batch_size, input_layer_dim]
        assert(c.shape[0] == batch_size)
        assert(len(c[0]) == input_layer_dim)
        preds = torch.empty((batch_size, next_layer_dim), device=self.device)
        for j in range(batch_size):
            # Forward for all neurons for sample j
            w_j = self.W[l_idx][:, c[j], range(input_layer_dim)]  # [next_layer_dim, input_layer_dim]
            preds[j] = w_j.matmul(x[j])
            # Update for sample j
            t, lr = 0, 0.1
            w_j = torch.clamp(w_j - lr * (logit_geo_mix(x[j], w_j) - t) * logit(x[j]),
                              min=-weight_clip, max=weight_clip)
            self.W[l_idx][:, c[j], range(input_layer_dim)] = w_j
        return preds

    def base_layer(self, batch_size, layer_size):
        clip_param = self.hparams["pred_clipping"]
        # rand_activations = torch.empty(batch_size, self.l_sizes[0]).normal_(mean=0.5, std=0.1)
        # rand_activations = torch.empty(batch_size, self.l_sizes[0]).normal_(mean=0.5, std=1.0).cuda()
        rand_activations = torch.empty((batch_size, self.l_sizes[0]), device=self.device).normal_(mean=0.5, std=1.0)
        rand_activations.requires_grad = False
        x = self.W_base(rand_activations)
        # x = torch.clamp(x, min=clip_param, max=1.0-clip_param)
        x = logit(x)
        return x

    def forward(self, s, y):
        """Calculate output of Gated Linear Network for input x and side info s

        Args:
            # x ([Float * [batch_size, input_layer_dim]]): Batch of arbitrary
            #                                              input activations
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
            y ([Float * [batch_size]]): Batch of binary targets

        Returns:
            [Float * [batch_size]]: Batch of GLN outputs (0 < probability < 1)
        """
        batch_size = s.shape[0]
        # x = x.view(batch_size, -1)
        # Base predictor followed by gated layers
        x = self.base_layer(batch_size, self.l_sizes[0])
        x = self.gated_layer(x, s, 0)
        x = self.gated_layer(x, s, 1)
        x = self.gated_layer(x, s, 2)
        return torch.sigmoid(x)

### OLD alternatives for gated_layer
# W_ctx = torch.empty((batch_size, next_layer_dim, input_layer_dim), device=self.device)
# for j in range(batch_size):
#     # For each sample:
#     # Select weights c[j] for each input neuron with contexts c[j, :]
#     W_ctx[j] = w[:, c[j], range(input_layer_dim)]  # [next_layer_dim, input_layer_dim]
### Below: potentially faster alternative to above loop
# W_ctx = torch.stack([w[:, c[j], range(input_layer_dim)] for j in range(batch_size)])

### Three alternatives below
# preds = torch.einsum('abc,ac->ab', W_ctx, x)  # TODO optimize using something other than einsum
# preds = torch.stack([W_ctx[i, :, :].matmul(x[i]) for i in range(batch_size)])  # Seems fastest on laptop CPU
# preds = torch.bmm(W_ctx, x.unsqueeze(2)).squeeze()
###