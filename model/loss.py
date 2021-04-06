import torch
import torch.nn.functional as F


def nll_loss(output, target):
    # print(output.is_cuda, target.is_cuda)
    # print(torch.sum(torch.isnan(output)), torch.sum(torch.isnan(target)))
    # return F.nll_loss(output, target)
    # loss = F.cross_entropy(output, target)
    # return loss
    predictions = torch.argmax(output, dim=1)
    num_incorrect = output.shape[0] - torch.sum(predictions == target)
    return num_incorrect
