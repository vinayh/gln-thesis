import torch


def get_params(s_dim, layer_size, num_subcontexts):
    """Initialize half-space context layer of specified size and num contexts

    Args:
        s_dim (Int): Size of side info s
        layer_size (Int): Size of layer to which contexts are applied (output dim)
        num_subcontexts (Int): Number of half-space subcontexts
                                i.e. num_contexts = 2**num_subcontexts

    Returns:
                [Float * [num_subctx, s_dim, layer_size]]: Weights of hyperplanes
    """
    ctx_weights = torch.empty(num_subcontexts, s_dim,
                              layer_size).normal_(mean=0, std=1.0)
    ctx_weights[:, -1, :].normal_(mean=0, std=0.5)
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
    subctx_sign = 0.5 * (torch.sign(calc_raw(s, ctx_weights, bitwise_map)) + 1)

    if gpu:
        return subctx_sign.float().matmul(bitwise_map.float()).long()
    else:
        return subctx_sign.long().matmul(bitwise_map)


def calc_raw(s, ctx_weights, bitwise_map):
    """Calculates subcontext distances for half-space gating given side info s

    Args:
        s ([Float * [batch_size, s_dim]]): Batch of side info samples s

    Returns:
        [Int * [batch_size, layer_size]]: Context indices for each side
                                            info sample in batch
    """
    ctx_dist = torch.matmul(s.expand(ctx_weights.shape[0],
                                     s.shape[0],
                                     s.shape[1]),
                            ctx_weights)
    ctx_results = ctx_dist.permute(1, 2, 0)
    # ctx_results = (torch.einsum('abc,db->dca', self.ctx_weights, s) > 0)
    return ctx_results
