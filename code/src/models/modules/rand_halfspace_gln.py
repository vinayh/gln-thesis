import torch
from torch import nn
from pytorch_lightning import LightningModule


class RandHalfSpaceGLN(LightningModule):
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
        self.register_buffer("ctx_weights", torch.empty(num_subcontexts, s_dim, layer_size).normal_(mean=0.5, std=1.0))

    def calc(self, s, gpu=False):
        """Calculates context indices for half-space gating given side info s

        Args:
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
            gpu (bool): Indicates whether model is running on GPU

        Returns:
            [Int * [batch_size, layer_size]]: Context indices for each side info
                                              sample in batch
        """
        # ctx_results = (torch.einsum('abc,db->dca', self.ctx_weights, s) > 0)
        ctx_dist = torch.matmul(s.expand(self.n_subctx, s.shape[0], s.shape[1]),
                                self.ctx_weights)
        ctx_results = (ctx_dist > 0).permute(1,2,0)
        if gpu:
            contexts = ctx_results.float().matmul(self.bitwise_map.float()).long()
        else:
            contexts = ctx_results.long().matmul(self.bitwise_map)
        return contexts