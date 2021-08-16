import torch


def logit(x):
    if torch.is_tensor(x):
        return torch.log(x / (1 - x))
    else:
        return torch.log(torch.tensor(x / (1 - x)))


def logit_geo_mix(logit_prev_layer, weights):
    """Geometric weighted mixture of prev_layer using weights

    Args:
        logit_prev_layer ([Float] * n_neurons): Logit of prev layer activations
        weights ([Float] * n_neurons): Weights for neurons in prev layer

    Returns:
        [Float]: sigmoid(w * logit(prev_layer)), i.e. output of current neuron
    """
    return torch.sigmoid(weights.matmul(logit_prev_layer))


def inv_sigmoid(x):
    return torch.log(x / (1 - x))


def to_one_vs_all(targets, num_classes, device="cpu"):
    """[summary]

    Args:
        targets ([type]): [description]
        num_classes ([type]): [description]
        device (str, optional): [description]. Defaults to 'cpu'.

    Returns:
        [type]: [description]
    """
    ova_targets = torch.zeros(
        (num_classes, len(targets)), dtype=torch.int, device=device
    )
    for i in range(num_classes):
        ova_targets[i, :][targets == i] = 1
    return ova_targets


# class STEFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return (input > 0).bool()

#     @staticmethod
#     def backward(ctx, grad_output):
#         return torch.nn.functional.hardtanh(grad_output)


# class StraightThroughEstimator(torch.nn.Module):
#     def __init__(self):
#         super(StraightThroughEstimator, self).__init__()

#     def forward(self, x):
#         x = STEFunction.apply(x)
#         return x


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.heaviside(
            input, torch.zeros(1).type_as(input)
        )  # this outputs 0 or 1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# 0.5*(Binary.apply(x) + 1)


def nan_inf_in_tensor(x):
    return torch.sum(torch.isinf(x)) + torch.sum(torch.isnan(x)) > 0
