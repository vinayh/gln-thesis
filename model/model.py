import torch
import torch.nn as nn
# import torch.nn.functional as F
from base import BaseModel


def logit(x):
    return torch.log(x / (torch.ones_like(x) - x))


def geo_mixture(weights, prev_layer):
    """Geometric weighted mixture of prev_layer using weights

    Args:
        weights ([Float] * num_neurons): Weights for neurons in prev layer
        prev_layer ([Float] * num_neurons): Activations of prev layer

    Returns:
        [Float]: sigmoid(w * logit(prev_layer)), i.e. output of current neuron
    """
    return torch.sigmoid(torch.dot(weights, logit(prev_layer)))


class GLNModel(BaseModel):
    """
    Binary GLN model, can be wrapped by GLNOneVsAllModel for multiclass tasks
    """
    n_context_fn = 4
    c_idx = None
    t = 1

    def __init__(self, n_context_fn=4, side_info_dim=784):
        super().__init__()
        self.n_context_fn = n_context_fn  # Num subcontext functions
        self.init_size = 200  # Num components in init arbitrary activations
        # self.l_size = [2000, 1000, 500, 1]
        self.l_size = [200, 100, 50, 1]  # Num neurons in each layer
        self.n_layers = len(self.l_size)
        self.l_out = [None] * (self.n_layers + 1)
        self.gw = [None] * self.n_layers
        self.gc = [None] * self.n_layers
        self.c_idx = [None] * self.n_layers

        for i, size in enumerate(self.l_size):
            self.gc[i] = self.ctx_init(n_context_fn, side_info_dim, size)
            if i == 0:
                self.gw[i] = self.weights_init(self.init_size, size)
            else:
                self.gw[i] = self.weights_init(self.l_size[i-1], size)

    def forward(self, s):
        """GLN forward pass steps:
        1. For each neuron, calculate contexts given input s (side info)
        2. For each neuron, identify the weights in these contexts
        3. For each neuron, find input activations
        4. For each neuron, calculate geometric mixture of input activations

        Args:
            s ([input sample] * batch_size): input features, used as side info

        Returns:
            [type]: [description]
        """
        print('Forward step of binary GLN')
        # Shape is [num_samples, num_neurons]
        self.l_out[0] = torch.rand(s.shape[0], self.init_size)
        s = torch.flatten(s, start_dim=2)
        self.l_out[1] = self.gated_layer(0, s)
        self.l_out[2] = self.gated_layer(1, s)
        self.l_out[3] = self.gated_layer(2, s)
        self.l_out[4] = self.gated_layer(3, s)
        return self.l_out[4][:, 0]

    def backward(self, targets, s=None):
        """Backward GLN step, updates appropriate weights using provided
        ground-truth targets along with either:
        1. last-used/saved input samples and contexts, or
        2. optionally provided input samples (and newly calculated contexts)
        TODO:
        1. For each sample/target, find ctx indices
        2. For each layer/neuron, update w for the correct ctx given target

        Args:
            targets ([target_label] * batch_size): categorical labels for batch
            s ([input_sample] * batch_size): OPTIONAL, input features

        Returns:
            Updated weights?
        """
        print('Backward step of binary GLN')
        eta = min(5500/self.t, 0.4)
        self.t += 1
        print(eta)
        for i in range(self.n_layers):  # For each layer's output
            output, prev_output = self.l_out[i+1], self.l_out[i]
            if s:  # If new samples are provided, calculate contexts
                c_idx = self.calc_contexts(self.gc[i], s)
            elif self.c_idx is not None:  # Saved contexts are used if found
                c_idx = self.c_idx[i]  # Layer ctx: [num_samples, num_neurons]
            else:  # If no samples and no saved contexts
                raise TypeError
            n_samples, n_neurons = output.shape
            assert(output.shape == c_idx.shape)  # [num_samples, num_neurons]
            for n in range(n_samples):  # For each sample
                # print('prev_output shape for sample n', prev_output.shape)
                for j in range(n_neurons):
                    c = c_idx[n, j]
                    # print('prev output shape', prev_output.shape)
                    tmp_1 = geo_mixture(self.gw[i][j, c, :], prev_output[n])
                    tmp_2 = logit(prev_output[n])
                    self.gw[i][j, c, :] -= eta * tmp_1 * tmp_2
                    # raise NotImplementedError

    def calc_contexts(self, ctx_params, s):
        """Calculates context indices for each input (side info) sample

        Args:
            ctx_params ([Float] * [side_info_dim,
                                   curr_layer_dim,
                                   n_subcontext]): Weights
            s ([input_sample] * batch_size): input features, i.e. side info

        Returns:
            [ctx_index * [num_samples, num_neurons]]: context ID in
                                                      0...2**num_subcontexts
        """
        num_samples = s.shape[0]
        num_neurons = ctx_params.shape[1]
        num_subcontexts = ctx_params.shape[2]
        # ctx_out = torch.zeros((num_samples, num_neurons), dtype=torch.int8)
        ctx_out = torch.zeros((num_samples, num_neurons), dtype=torch.int8, device='cuda:0')
        for c in range(num_subcontexts):
            for n in range(num_samples):
                # assert(ctx_out[n, :].shape == torch.matmul(s[n, 0, :], ctx_params[:, :, c]).shape)
                ctx_out[n, :] += (2**c * (torch.matmul(s[n, 0, :], ctx_params[:, :, c]) > 0))
        return ctx_out

    def gated_layer(self, l_idx, s):
        """Gated layer operation

        Args:
            # ctx_params ([Float] * [side_info_dim,
            #                        curr_layer_dim,
            #                        n_subcontext]): Context params
            # weight_params ([Float] * [curr_layer_dim,
            #                           num_contexts,
            #                           prev_layer_dim]): Weights
            # prev_layer ([type]): [description]
            # s ([type]): [description]

        Returns:
            [type]: [description]
        """
        ctx_params = self.gc[l_idx]  # [side_info_dim, curr_l_dim, n_subctx]
        weight_params = self.gw[l_idx]  # [curr_l_dim, n_contexts, prev_l_dim]
        prev_layer = self.l_out[l_idx]  # [prev_l_dim]

        c_idx = self.calc_contexts(ctx_params, s)  # [num_samples, num_neurons]
        self.c_idx[l_idx] = c_idx
        # Gated weights for each sample and each neuron
        # Shape is [num_samples, num_neurons, weight_components]
        num_samples, num_neurons = c_idx.shape[:2]
        output = torch.empty(num_samples, num_neurons)
        # ############# This is the slow part!
        for i in range(num_samples):
            for j in range(num_neurons):
                output[i, j] = geo_mixture(weight_params[j, c_idx[i, j], :],
                                           prev_layer[i])
        # output = torch.tensor([[geo_mixture(weight_params[j, c_idx[i, j], :],
        #                                     prev_layer[i])
        #                         for j in range(num_neurons)]
        #                        for i in range(num_samples)])
        return output

    def ctx_init(self, n_subcontext, side_info_dim, curr_layer_dim):
        """Creates linear layer representing a context function
        for a provided previous and current layer size

        Args:
            n_subcontext (Float): Number of subcontext functions
            side_info_dim (Float): Size of side info
            curr_layer_dim (Float): Size of current layer

        Returns:
            # nn.Linear: Context function for specified layer sizes
            [Float] * [side_info_dim, curr_layer_dim, n_subcontext]: Weights
        """
        ctx_params = torch.normal(mean=0.0, std=0.1, size=(side_info_dim,
                                                           curr_layer_dim,
                                                           n_subcontext))
        # torch.nn.init.normal_(ctx_params, mean=0.0, std=0.1)
        # c = nn.Linear(side_info_dim, curr_layer_dim)
        # # Is this the correct stdev from the paper?
        # nn.init.normal_(c.weight, mean=0.0, std=0.1)
        return ctx_params

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
                              prev_layer_dim))
        return weights / prev_layer_dim  # Need to be init to 1/prev_layer_dim
