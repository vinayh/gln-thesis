import torch
from torch import nn


class HalfSpace(nn.Module):
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
        self.bitwise_map = torch.tensor([2**i for i in range(self.n_subctx)])
        # Init subcontext functions (half-space gatings)
        self.subctx_fn = []
        for _ in range(self.n_subctx):
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
        ctx = torch.zeros((s.shape[0], self.subctx_fn[0].out_features),
                          dtype=torch.int8)
        for i in range(self.n_subctx):
            ctx += (self.subctx_fn(s) > 0) * self.bitwise_map[i]
        return ctx

class GLNLayers(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.ctx = []
        self.W = []
        l_sizes = (hparams["lin1_size"], hparams["lin2_size"], hparams["lin3_size"])
        # Weights for base predictor (arbitrary activations)
        self.W_base = torch.rand((l_sizes[0], l_sizes[0]), 1.0/l_sizes[0])
        for i in range(len(l_sizes)-1):
            self.ctx.append(HalfSpace(hparams["input_size"],
                                      l_sizes[i],
                                      hparams["num_subcontexts"]))
            self.W.append(torch.full((l_sizes[i+1],
                                      2**hparams["num_subcontexts"],
                                      l_sizes[i]),
                                     1.0/l_sizes[i]))
        # self.model = nn.Linear(hparams["input_size"], hparams["lin1_size"])

    def get_contexts(self, hparams: dict):
        return

    def forward(self, x, s):
        """Calculate output of Gated Linear Network for input x and side info s

        Args:
            x ([Float * [batch_size, input_layer_dim]]): Batch of arbitrary
                                                         input activations
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s

        Returns:
            [Float * [batch_size]]: Batch of GLN outputs (0 < probability < 1)
        """
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)
