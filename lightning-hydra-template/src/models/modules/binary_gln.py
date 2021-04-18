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
        self.layer_size = layer_size
        self.bitwise_map = torch.tensor([2**i for i in range(self.n_subctx)], device=self.device)
        # Init subcontext functions (half-space gatings)
        self.subctx_fn = []
        for _ in range(self.n_subctx):
            ### GPU
            # new_subctx = nn.Linear(s_dim, layer_size).cuda()
            ### CPU
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
        ctx = torch.zeros((batch_size, self.layer_size),
                          dtype=torch.long, device=self.device)
        for i in range(self.n_subctx):
            ctx += (self.subctx_fn[i](s.flatten(start_dim=2)) > 0).squeeze(dim=1) * self.bitwise_map[i]
        return ctx

class BinaryGLN(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        self.ctx = []
        self.W = []
        self.lr = 0.4
        self.t = 1
        self.l_sizes = (hparams["lin1_size"], hparams["lin2_size"], hparams["lin3_size"], 1)
        # Weights for base predictor (arbitrary activations)
        # self.W_base = torch.rand((self.l_sizes[0], self.l_sizes[0]), 1.0/self.l_sizes[0])
        self.W_base = nn.Linear(self.l_sizes[0], self.l_sizes[0])
        self.W_base.requires_grad_ = False
        # Context functions and weights for gated layers
        s_dim = hparams["input_size"]
        for i in range(len(self.l_sizes)-1):  # Add ctx and w for each layer until the single output neuron layer
            curr_layer_dim, next_layer_dim = self.l_sizes[i], self.l_sizes[i+1]
            ### GPU
            # self.ctx.append(HalfSpace(s_dim,
            #                           curr_layer_dim,
            #                           hparams["num_subcontexts"]).cuda())
            # self.W.append(torch.full((next_layer_dim,
            #                           2**hparams["num_subcontexts"],
            #                           curr_layer_dim),
            #                          1.0/curr_layer_dim).cuda())
            ### CPU
            self.ctx.append(HalfSpace(s_dim,
                                      curr_layer_dim,
                                      hparams["num_subcontexts"]))
            self.W.append(torch.full((next_layer_dim,
                                      2**hparams["num_subcontexts"],
                                      curr_layer_dim),
                                     1.0/curr_layer_dim))
    
    def gated_layer(self, logit_x, s, y, l_idx):
        """Using provided input activations, context functions, and weights,
           returns the result of the GLN layer

        Args:
            logit_x ([Float * [batch_size, input_layer_dim]]): Input activations (with logit applied)
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
            y ([Float * [batch_size]]): Batch of binary targets
            l_idx (Int): Index of layer to use for selecting correct ctx and layer weights
        Returns:
            [Float * [batch_size, next_layer_dim]]: Output of GLN layer
        """
        batch_size = s.shape[0]
        next_layer_dim, _, input_layer_dim = self.W[l_idx].shape
        w_clip = self.hparams["weight_clipping"]
        assert(self.W[l_idx].shape[2] == logit_x.shape[1])
        assert(logit_x.shape[0] == batch_size)
        c = self.ctx[l_idx].calc(s)  # [batch_size, input_layer_dim]
        assert(c.shape[0] == batch_size)
        assert(len(c[0]) == input_layer_dim)
        logit_preds = torch.empty((batch_size, next_layer_dim), device=self.device)
        for j in range(batch_size):
            # Forward for all neurons for sample j
            w_j = self.W[l_idx][:, c[j], range(input_layer_dim)]  # [next_layer_dim, input_layer_dim]
            logit_preds[j] = w_j.matmul(logit_x[j])
            # Update for sample j
            # w_j_delta = (logit_geo_mix(logit_x[j], w_j) - y[j]).matmul(logit_x[j])
            loss = torch.sigmoid(logit_preds[j]) - y[j]  # [next_layer_dim]: Loss for each output neuron
            w_j_delta = torch.outer(loss, logit_x[j])  # [next_layer_dim, input_layer_dim]
            if torch.any(torch.isnan(w_j_delta)):
                raise Exception
            w_j = torch.clamp(w_j - self.lr * w_j_delta,
                              min=-w_clip, max=w_clip)
            self.W[l_idx][:, c[j], range(input_layer_dim)] = w_j
        return logit_preds

    def base_layer(self, batch_size, layer_size):
        clip_param = self.hparams["pred_clipping"]
        ### GPU
        # rand_activations = torch.empty(batch_size, self.l_sizes[0]).normal_(mean=0.5, std=1.0).cuda()
        ### CPU
        rand_activations = torch.empty(batch_size, self.l_sizes[0]).normal_(mean=0.5, std=0.1)
        rand_activations.requires_grad = False
        x = self.W_base(rand_activations)
        x = torch.clamp(x, min=clip_param, max=1.0-clip_param)
        return logit(x)

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
        self.t += batch_size
        self.lr = min(0.4, 5000/self.t)
        with torch.no_grad():
            # x = x.view(batch_size, -1)
            # Base predictor followed by gated layers
            x = self.base_layer(batch_size, self.l_sizes[0])
            x = self.gated_layer(x, s, y, 0)
            x = self.gated_layer(x, s, y, 1)
            x = self.gated_layer(x, s, y, 2)
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