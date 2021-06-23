import torch
from pytorch_lightning import LightningModule

from src.utils.helpers import STEFunction
from src.utils.helpers import Binary


class RandHalfSpaceDGN(LightningModule):
    def __init__(self, num_features, layer_size, num_branches, ctx_bias=True, trained_ctx=False):
        """Initialize half-space context layer of specified size and num contexts

        Args:
            num_features (Int): Size of side info s, excluding bias
            layer_size (Int): Size of layer to which contexts are applied (output dim)
            num_branches (Int): Number of half-space gating branches
        """
        super().__init__()
        self.register_buffer("hyperplanes", torch.empty(
            layer_size, num_branches, num_features).normal_(mean=0, std=1.0))
        # self.hyperplanes == torch.nn.parameters([layer_size, num_branches,
        #                                num_features])
        self.hyperplanes = self.hyperplanes / torch.linalg.norm(
            self.hyperplanes[:, :, :-1], axis=(1, 2))[:, None, None]
        self.hyperplanes[:, :, -1].normal_(mean=0, std=0.5)
        self.hyperplanes.requires_grad = True
        self.hyperplanes = torch.nn.Parameter(self.hyperplanes,
                                              requires_grad=True)

    def calc(self, s, gpu=False):
        """Calculates context indices for half-space gating given side info s

        Args:
            s ([Float * [batch_size, num_features]]): Batch of side info samples s
            gpu (bool): Indicates whether model is running on GPU

        Returns:
            [Int * [batch_size, layer_size]]: Context indices for each side info
                                              sample in batch
        """
        ste = Binary.apply  # Straight-through estimator function
        return 0.5*(ste(torch.einsum('abc,dc->dba', self.hyperplanes, s))+1)
        # return (torch.einsum('abc,dc->dba', self.hyperplanes, s) > 0).bool()
        # return (self.hyperplanes.matmul(s.permute(1, 0)).permute(2, 1, 0) > 0).bool()

    def calc_raw(self, X_all):
        """Calculate raw values for plotting decision boundary

        Args:
            X_all ([Float * [batch_size, num_features]]): All input samples X
        """
        # return self.hyperplanes.matmul(X_all.permute(1, 0)).permute(2, 1, 0)
        return torch.einsum('abc,dc->dba', self.hyperplanes, X_all)
