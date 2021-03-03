import torch
import torch.nn as nn
# import torch.nn.functional as F
from base import BaseModel


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
    # return torch.sigmoid(torch.dot(weights, logit(prev_layer)))
    tmp_1 = torch.prod(torch.pow(prev_layer, weights))
    tmp_2 = torch.prod(torch.pow(1 - prev_layer, weights))
    return tmp_1 / (tmp_1 + tmp_2)



class GLNModel(BaseModel):
    """
    Binary GLN model, can be wrapped by GLNOneVsAllModel for multiclass tasks
    """
    n_context_fn = 4
    contexts = None
    t = 1
    init_size = 200  # Num components in init arbitrary activations
    # l_size = [2000, 1000, 500, 1]
    l_size = [200, 100, 50, 1]  # Num neurons in each layer

    def __init__(self, n_context_fn=4, side_info_dim=784, ctx_type='half_space'):
        super().__init__()
        self.n_context_fn = n_context_fn  # Num subcontext functions
        self.n_layers = len(self.l_size)
        self.ctx_type = ctx_type
        self.l_out = [None] * (self.n_layers + 1)
        self.gw = [None] * self.n_layers  # [n_layers, n_samples, n_neurons, w]
        self.gc = [None] * self.n_layers  # [n_layers, side_info_dim, curr_l_dim, n_subctx]
        self.c = [None] * self.n_layers  # [n_layers, n_samples, n_neurons]

        # Init weights and context params
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

    def calc_contexts(self, ctx_fn, s):
        """Calculates context indices for each input (side info) sample

        Args:
            ctx_fn ([nn.Linear] * [curr_layer_dim,
                                   n_subcontext]): Context functions
            s ([Float] * [batch_size,
                          side_info_dim]): Input features, i.e. side info

        Returns:
            [ctx_index * [n_samples, n_neurons]]: context ID in
                                                      0...2**num_subcontexts
        """
        # num_subcontexts = ctx_params.shape[2]
        # # ctx_out = torch.zeros((n_samples, n_neurons), dtype=torch.int8)
        # ctx_out = torch.zeros((n_samples, n_neurons),
        #                       dtype=torch.int8,
        #                       device='cuda')
        # for c in range(num_subcontexts):
        #     tmp = (s[:, 0, :].matmul(ctx_params[:, :, c]) > 0)
        #     assert(tmp.shape == ctx_out.shape)
        #     ctx_out += 2**c * tmp
        n_samples = s.shape[0]
        n_neurons = len(ctx_fn)
        ctx_out = torch.zeros((n_samples, n_neurons),
                              dtype=torch.int8,
                              device='cuda')
        for n in range(n_neurons):
            ctx = ctx_fn[n] # Context functions for given layer and neuron n
            tmp_2 = torch.empty((n_samples, len(ctx)), device='cuda')
            for i in range(n_samples):
                for c in range(len(ctx)):
                    tmp_1 = torch.sign(ctx[c](s[i, :]))
                    tmp_2[:, c] = 2**c * (tmp_1 > 0)
                ctx_out[i, n] = torch.sum(tmp_2, dim=1)
        return ctx_out

    def gated_layer(self, l_idx, s):
        """Gated layer operation

        Args:
            # prev_layer ([type]): [description]
            # s ([type]): input features, i.e. side info

        Returns:
            [type]: [description]
        """
        self.c[l_idx] = self.calc_contexts(self.gc[l_idx], s)
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
        # ctx_params = torch.normal(mean=0.0,
        #                           std=0.1,
        #                           size=(side_info_dim,
        #                                 curr_layer_dim,
        #                                 n_subcontext),
        #                           device='cuda')
        # torch.nn.init.normal_(ctx_params, mean=0.0, std=0.1)
        if self.ctx_type == 'half_space':
            all_ctx_functions = []
            for _ in range(curr_layer_dim):
                tmp_ctx_functions = []
                for _ in range(n_subcontext):
                    c = nn.Linear(side_info_dim, curr_layer_dim)
                    nn.init.normal_(c.weight, mean=0.0, std=0.1) # TODO: check stdev in paper
                    tmp_ctx_functions.append(c)
                all_ctx_functions.append(tmp_ctx_functions)
        else:
            raise Exception
        return all_ctx_functions

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

"""
Code in forward() if we use a tmp layer output placeholder to save last batch
    self.tmp_l_out = [None] * len(self.l_out)
    for i in range(self.n_layers):
        self.tmp_l_out[i+1] = self.gated_layer(i, s)
    for i in range(self.n_layers):
        self.l_out[i+1] = self.tmp_l_out[i+1]
    del self.tmp_l_out
    torch.cuda.empty_cache()
"""

"""Stuff from gated_layer()

# if self.l_out[l_idx] is not None:  # If output from prev batch exists
        #     prev_layer = self.l_out[l_idx]  # [prev_l_dim]
        # else:  # If layer output has never been saved
        #     prev_layer = self.tmp_l_out[l_idx]


for j in range(n_neurons):
                c = self.c[l_idx][i, j]
                # TODO: Many values here end up at 1.0 which causes a NaN later
                output[i, j] = geo_mix(prev_layer[i], self.gw[l_idx][j, c, :])
"""