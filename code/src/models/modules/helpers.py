import torch

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
    return torch.sigmoid(weights.matmul(logit_prev_layer))