import torch
import torch.nn as nn
from base import BaseModel
from model import GLNModel

def to_one_vs_all(targets):
    """
    Input: Torch tensor of target values (categorical labels)
    Returns: List of Torch tensors containing one-hot targets
                for each class (used for one-vs-all models)
    """
    classes = torch.unique(targets, sorted=True)
    ova_targets = torch.zeros((len(classes), len(targets)),
                              dtype=torch.int,
                              device='cuda')
    for i, c in enumerate(classes):
        ova_targets[i, :][targets == c] = 1
    return ova_targets


class GLNOneVsAllModel(BaseModel):
    """
    One-vs-all model wrapping the GLNModel binary classifier
    """
    def __init__(self, n_classes=10, n_context_fn=4, prev_layer_dim=784):
        super(GLNOneVsAllModel, self).__init__()
        self.n_context_fn = n_context_fn
        self.n_classes = n_classes
        # Create a binary model for each class (i.e. each one-vs-all model)
        self.models = [GLNModel(n_context_fn, prev_layer_dim)
                       for i in range(n_classes)]
        for m in self.models:
            m.to(torch.device('cuda'))

    def forward(self, s):
        """Forward pass of GLN one-vs-all model
        Targets do not need to be converted from categorical to one-vs-all

        Args:
            s ([input sample] * batch_size): input features, used as side info

        Returns:
            [type]: [description]
        """
        output = torch.stack([m.forward(s) for m in self.models]).cuda()
        # predictions = torch.argmax(output, dim=0)
        print(output.T.shape)
        print
        return output.T  # transpose so new shape is [num_samples, num_classes]

    def backward(self, targets):
        """Backward pass of GLN one-vs-all model
        Targets are converted to one-vs-all and fed into
        each one-vs-all model

        Args:
            targets ([target_label] * batch_size): categorical labels for batch
        """
        ova_targets = to_one_vs_all(targets)
        for i, model in enumerate(self.models):
            model.backward(ova_targets[i])
