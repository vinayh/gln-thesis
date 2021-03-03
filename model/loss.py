import torch
import torch.nn.functional as F


def nll_loss(output, target):
    # print(output.is_cuda, target.is_cuda)
    # print(torch.sum(torch.isnan(output)), torch.sum(torch.isnan(target)))
    # return F.nll_loss(output, target)
    loss = F.cross_entropy(output, target)
    print(loss)
    return loss
