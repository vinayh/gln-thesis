import torch

from src.utils.helpers import StraightThroughEstimator


def get_params(num_features, layer_size, num_branches, ctx_bias=True, pretrained_ctx=False, device=None):
    """Return weights for half-space context layer of specified size and num_contexts

        Args:
            num_features (Int): Size of side info s, excluding bias
            layer_size (Int): Size of layer to which contexts are applied (output dim)
            num_branches (Int): Number of half-space gating branches

        Returns:
            [Float * [layer_size, num_branches, num_features]]: Weights of hyperplanes
    """
    hyperplanes = torch.empty(layer_size, num_branches, num_features, device=device)
    # hyperplanes = hyperplanes / torch.linalg.norm(
    #     hyperplanes[:, :, :-1], axis=(1, 2))[:, None, None]
    hyperplanes.normal_(mean=0, std=1.0)
    hyperplanes[:, :, -1].normal_(mean=0, std=0.5)
    return hyperplanes


def calc(s, hyperplanes, gpu=False):
    """Calculates context indices for half-space gating given side info s

    Args:
        s ([Float * [batch_size, num_features]]): Batch of side info samples s
        hyperplanes ([Float * [layer_size, num_branches, num_features]]): Weights of hyperplanes
        gpu (bool): Indicates whether model is running on GPU

    Returns:
        [Int * [batch_size, layer_size]]: Context indices for each side info
                                            sample in batch
    """
    # product = torch.einsum('abc,dc->dba', hyperplanes, s)
    # with torch.no_grad():
    product = hyperplanes.matmul(s.T).permute(2, 1, 0)

    # Straight-through estimator
    return StraightThroughEstimator.apply(product)
    # return (torch.einsum('abc,dc->dba', hyperplanes, s) > 0).bool()
    # return (hyperplanes.matmul(s.permute(1, 0)).permute(2, 1, 0) > 0).bool()


def calc_raw(X_all, hyperplanes):
    """Calculate raw values for plotting decision boundary

    Args:
        X_all ([Float * [batch_size, num_features]]): All input samples X
        hyperplanes ([Float * [layer_size, num_branches, num_features]]): Weights of hyperplanes
    """
    return hyperplanes.matmul(X_all.permute(1, 0)).permute(2, 1, 0)
    # return torch.einsum('abc,dc->dba', hyperplanes, X_all)
