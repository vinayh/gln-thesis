import torch

def logit(x):
    return torch.log(x / (torch.ones_like(x) - x))

def geo_mix(prev_layer, weights):
    """Geometric weighted mixture of prev_layer using weights

    Args:
        weights ([Float] * n_neurons): Weights for neurons in prev layer
        prev_layer ([Float] * n_neurons): Activations of prev layer

    Returns:
        [Float]: sigmoid(w * logit(prev_layer)), i.e. output of current neuron
    """
    # if weights.isnan().any():
    #     raise Exception
    return torch.sigmoid(torch.dot(weights, logit(prev_layer)))
    # tmp_1 = torch.prod(torch.pow(prev_layer, weights))
    # tmp_2 = torch.prod(torch.pow(1 - prev_layer, weights))
    # return tmp_1 / (tmp_1 + tmp_2)