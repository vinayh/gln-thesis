import torch
# import torch.nn as nn
from base import BaseModel
from .helpers import logit, geo_mix
from .context_halfspace import HalfSpace

class GLNModel(BaseModel):
    """
    Binary GLN model, can be wrapped by GLNOneVsAllModel for multiclass tasks
    """
    n_context_fn = 4
    t = 1
    init_size = 200  # Num components in init arbitrary activations
    # l_size = [2000, 1000, 500, 1]
    l_size = [200, 100, 50, 1]  # Num neurons in each layer

    def __init__(self, n_context_fn=4, side_info_dim=784, ctx_type='half_space'):
        super().__init__()
        self.n_layers = len(self.l_size)
        self.l_out = [None] * (self.n_layers + 1)
        self.gw = [None] * self.n_layers  # [n_layers, n_samples, n_neurons, w]
        self.gc = [None] * self.n_layers  # [n_layers, side_info_dim, curr_l_dim, n_subctx]
        self.c = [None] * self.n_layers  # [n_layers, n_samples, n_neurons]

        if ctx_type == 'half_space':
            self.ctx = HalfSpace()
        else:
            self.ctx = None

        # Init weights and context params
        for i, layer_size in enumerate(self.l_size):
            self.gc[i] = self.ctx.init_fn(side_info_dim, layer_size)
            if i == 0:
                self.gw[i] = self.weights_init(self.init_size, layer_size)
            else:
                self.gw[i] = self.weights_init(self.l_size[i-1], layer_size)

    def forward(self, s):
        """GLN forward pass steps:
        1. For each neuron, calculate contexts given input s (side info)
        2. For each neuron, identify the weights in these contexts
        3. For each neuron, find input activations
        4. For each neuron, calculate geometric mixture of input activations

        Args:
            s ([input sample] * batch_size): input features, used as side info

        Returns:
            [type]: Result in output layer
        """
        print('Forward step of binary GLN')
        s = torch.flatten(s, start_dim=2).cuda()
        self.l_out[0] = torch.rand(s.shape[0], self.init_size, device='cuda')
        for i in range(self.n_layers):
            self.l_out[i+1] = self.gated_layer(i, s)
        if(self.l_out[-1].isnan().any()):
            raise Exception
        return self.l_out[-1][:, 0]

    def backward(self, targets, s=None):
        """Backward GLN step, updates appropriate weights using provided
        ground-truth targets along with either:
        1. last-used/saved input samples and contexts, or
        2. optionally provided input samples (and newly calculated contexts)

        Args:
            targets ([target_label] * batch_size): categorical labels for batch
            s ([input_sample] * batch_size): OPTIONAL, input features

        Returns:
            Updated weights?
        """
        print('Backward step of binary GLN')
        for i in range(self.n_layers):  # For each layer's output
            prev_output = self.l_out[i]  # Prev layer's output
            n_samples, n_neurons = self.l_out[i+1].shape  # This layer's output
            contexts = self.retrieve_contexts(s, i)  # [n_samples, n_neurons]
            for n in range(n_samples):  # For each sample
                eta = self.eta()
                for j in range(n_neurons):
                    c = contexts[n, j]
                    tmp_1 = geo_mix(prev_output[n], self.gw[i][j, c, :])
                    tmp_2 = logit(prev_output[n])
                    # TODO: Weight param vectors are assigned to scalars?
                    self.gw[i][j, c, :] -= eta * (tmp_1 - targets[n]) * tmp_2
    
    def eta(self):
        self.t += 1  # TODO: Fix how t is incremented
        eta = min(5500/self.t, 0.4)
        return eta
    
    def retrieve_contexts(self, s, i):
        """Gets contexts by calculating them from new samples, if provided,
        or from previously saved contexts if samples are not provided.

        Args:
            s ([input_sample] * batch_size): OPTIONAL, input features
            i (Float): Index of current layer

        Raises:
            TypeError: If no contexts are saved and no samples are provided

        Returns:
            [Float] * batch_size: IDs of contexts for each sample/neuron combo
        """
        if s:  # If new samples are provided, calculate contexts
            return self.calc_contexts(self.gc[i], s)
        elif self.c is not None:  # Saved contexts are used if found
            return self.c[i]  # Layer ctx: [n_samples, n_neurons]
        else:  # If no samples and no saved contexts
            raise TypeError


    def gated_layer(self, l_idx, s):
        """Gated layer operation

        Args:
            # prev_layer ([type]): [description]
            # s ([type]): input features, i.e. side info

        Returns:
            [type]: [description]
        """
        self.c[l_idx] = self.ctx.calc(self.gc[l_idx], s)
        n_samples, n_neurons = self.c[l_idx].shape[:2]
        prev_layer = self.l_out[l_idx]
        if prev_layer.isnan().any():
            raise Exception

        # Gated weights for each sample and each neuron
        output = torch.empty((n_samples, n_neurons), device='cuda')
        for n in range(n_samples):  # ############# This is the slow part!
            w_i = self.gw[l_idx]  # [n_samples, n_neurons, w]
            c = self.c[l_idx][n, :]  # [n_neurons]
            W = torch.stack([w_i[k, c[k], :] for k in range(n_neurons)])
            output[n, :] = torch.sigmoid(W.matmul(logit(prev_layer[n, :])))
            # print(torch.max(output), torch.min(output))
        if output.isnan().any():  # TODO: This NaN still occurs
            raise Exception
        return output

    def weights_init(self, prev_layer_dim, curr_layer_dim):
        """Generate initial uniform weights for each context in layer's neurons

        Args:
            prev_layer_dim (Float): Size of previous layer
            curr_layer_dim (Float): Size of current layer

        Returns:
            [Float] * [curr_layer_dim, num_contexts, prev_layer_dim]: Weights
        """
        weights = torch.ones((curr_layer_dim,
                              2**self.n_context_fn,
                              prev_layer_dim),
                              device='cuda')
        return weights / prev_layer_dim  # Need to be init to 1/prev_layer_dim
