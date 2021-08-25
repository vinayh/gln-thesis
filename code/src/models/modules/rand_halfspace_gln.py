import torch
from src.utils.helpers import StraightThroughEstimator


def get_params(hparams, layer_size):
    """Initialize half-space context layer of specified size and num contexts

    Args:
        s_dim (Int): Size of side info s
        layer_size (Int): Size of layer to which contexts are applied (output dim)
        num_subcontexts (Int): Number of half-space subcontexts
                                i.e. num_contexts = 2**num_subcontexts

    Returns:
                [Float * [num_classes, num_subctx, layer_size, s_dim]]: Weights of hyperplanes
    """
    with torch.no_grad():
        # TODO: bias
        s_dim = hparams["input_size"] + 1
        # s_dim = hparams["input_size"]
        # pretrained_ctx = hparams["pretrained_ctx"]
        ctx_weights = torch.empty(
            hparams["num_classes"], hparams["num_subcontexts"], layer_size, s_dim
        ).normal_(mean=0, std=1.0)
        if hparams["ctx_bias"]:
            ctx_weights[:, -1, :].normal_(mean=0, std=0.5)
        else:
            ctx_weights[:, -1, :] = 0
        return ctx_weights


def calc(s, ctx_weights, bitwise_map, gpu=False):
    """Calculates context indices for half-space gating given side info s

    Args:
        s ([Float * [batch_size, s_dim]]): Batch of side info samples s
        gpu (bool): Indicates whether model is running on GPU

    Returns:
        [Int * [batch_size, layer_size]]: Context indices for each side
                                            info sample in batch
    """
    # Get 0 or 1 based on sign of each input sample with respect to each subctx
    subctx_sign = StraightThroughEstimator.apply(calc_raw(s, ctx_weights))
    if gpu:
        return bitwise_map.float().matmul(subctx_sign).long()
    else:
        return bitwise_map.long().matmul(subctx_sign)


def calc_raw(s, ctx_weights):
    """Calculates subcontext distances for half-space gating given side info s

    Args:
        s ([Float * [batch_size, s_dim]]): Batch of side info samples s

    Returns:
        [Int * [batch_size, layer_size]]: Context indices for each side
                                            info sample in batch
    """
    # ctx_dist: [num_classes, num_subcontexts, layer_size]
    ctx_dist = torch.matmul(ctx_weights, s.squeeze(0))
    # ctx_results = ctx_dist.permute(1, 2, 0)
    # # ctx_results = (torch.einsum('abc,db->dca', self.ctx_weights, s) > 0)
    return ctx_dist
