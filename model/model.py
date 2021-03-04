import torch
import math
# import torch.nn as nn
from base import BaseModel
from .helpers import logit, geo_mix, logit_geo_mix
from .context_halfspace import HalfSpace

class GLNModel(BaseModel):
    """
    Binary GLN model, can be wrapped by GLNOneVsAllModel for multiclass tasks
    """
    n_context_fn = 4
    t = 1
    K = [2000, 2000, 1000, 500, 1]
    # K = [100, 10, 5, 2, 1]  # Num neurons in each layer including 0'th layer

    def __init__(self, n_context_fn=4, side_info_dim=784, ctx_type='half_space'):
        super().__init__()
        self.L = [None] * len(self.K)
        self.logit_L = [None] * len(self.K)
        self.ctx = [None] * len(self.K)  # [n_layers]
        self.gw = [None] * len(self.K)  # [n_layers, n_contexts, n_neurons, w]
        self.c = [None] * len(self.K)  # [n_layers, n_samples, n_neurons]

        # Init weights and context params
        self.gw[0] = self.weights_init(self.K[0], self.K[1])
        for i in range(1, len(self.K)):
            self.gw[i] = self.weights_init(self.K[i-1], self.K[i])
            if ctx_type == 'half_space':
                self.ctx[i] = HalfSpace()
                self.ctx[i].init_fn(side_info_dim, self.K[i])
            else:
                raise Exception

    def forward(self, s):
        """GLN forward pass steps:
        1. For each neuron, calculate contexts given input s (side info)
        2. For each neuron, identify the weights in these contexts
        3. For each neuron, find input activations
        4. For each neuron, calculate geometric mixture of input activations

        Args:
            s ([input sample] * batch_size): input features, used as side info

        Returns:
            [type]: Result in L_i layer
        """
        print('Forward step of binary GLN')
        s = torch.flatten(s, start_dim=2).cuda()
        # self.L[0] = torch.rand(s.shape[0], self.K[0], device='cuda')
        self.logit_L[0] = logit(torch.rand(s.shape[0], self.K[0], device='cuda'))
        for i in range(1, len(self.K)):
            # self.logit_L[i], self.L[i] = self.gated_layer(i, s)
            self.logit_L[i] = self.gated_layer(i, s)
        # self.L = self.gated_layers_optimized(s)  # Calc. all layers together

        # if self.L[-1].isnan().any():
        #     raise Exception
        if torch.sigmoid(self.logit_L[-1]).isnan().any():
            raise Exception
        # output = self.L[-1][:, 0]  # Output of last layer
        output = torch.sigmoid(self.logit_L[-1][:, 0])
        print('output', output)
        return output
    
    def update(self, eta, logit_L_prev_z, w_ijc, targets_z):
        delta = eta * (logit_geo_mix(logit_L_prev_z, w_ijc) - targets_z) * logit_L_prev_z
        new_w_ijc = torch.clamp(w_ijc - delta, min=-1.0, max=1.0)
        # if torch.max(new_w_ijc) > 0.99:
        #     raise Exception
        return new_w_ijc

    def backward(self, t, s=None):
        """Backward GLN step, updates appropriate weights using provided
        ground-truth targets along with either:
        1. last-used/saved input samples and contexts, or
        2. optionally provided input samples (and newly calculated contexts)

        Args:
            t ([target_label] * batch_size): target labels for batch
            s ([input_sample] * batch_size): OPTIONAL, input features

        Returns:
            Updated weights?
        """
        print('Backward step of binary GLN')
        # print('targets', t)
        for i in range(1, len(self.K)):  # For each non-base layer
            eta = self.eta()  # Update learning rate
            logit_L_prev = self.logit_L[i-1]  # Prev layer output
            n_samples, n_neurons = self.logit_L[i].shape
            contexts = self.retrieve_contexts(s, i)  # [n_samples, n_neurons]
            for z in range(n_samples):  # For each sample
                for j in range(n_neurons):
                    c_ijz = contexts[z, j]
                    w_ijc = self.gw[i][j, c_ijz, :]
                    self.gw[i][j, c_ijz, :] = self.update(eta, logit_L_prev[z], w_ijc, t[z])

    
    def eta(self):
        self.t += 1  # TODO: Fix how t is incremented
        n = min(5500/self.t, 0.4)
        # print('eta:', n)
        return n
    
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
            return self.ctx[i].calc(s)
        elif self.c is not None:  # Saved contexts are used if found
            return self.c[i]  # Layer ctx: [n_samples, n_neurons]
        else:  # If no samples and no saved contexts
            raise TypeError

    # def gated_layers_optimized(self, s):
    #     """Gated layer operation

    #     Args:
    #         # s ([type]): Input features, i.e. side info

    #     Returns:
    #         [type]: [description]
    #     """
    #     n_samples = len(s)  # Number of side info samples
    #     # values = [logit(self.L[0][z]) for z in range(n_samples)]
    #     self.L = [torch.empty((n_samples, K_i), device='cuda') for K_i in self.K]
    #     self.L[0] = torch.rand(s.shape[0], self.K[0], device='cuda')
    #     for z in range(n_samples):
    #         temp = logit(self.L[0][z])
    #         for i in range(1, len(self.K)):
    #             self.c[i] = self.ctx[i].calc(s)  # [n_samples, n_neurons]
    #             n_neurons = self.c[i].shape[1]
    #             W = torch.stack([self.gw[i][j, self.c[i][z][j], :] for j in range(n_neurons)]).to('cuda')
    #             temp = W.matmul(temp.T)
    #             # self.L[i][z, :] = torch.sigmoid(temp)
    #         self.L[-1][z, :] = torch.sigmoid(temp)
    #     # print('Max', torch.max(self.L[-1]), 'Min', torch.min(self.L[-1]), 'layer', i)
    #     # if self.L[-1].isnan().any():  # TODO: This NaN still occurs
    #     #     raise Exception
    #     return self.L

    def gated_layer(self, i, s):
        """Gated layer operation

        Args:
            # i (int): Index of current layer
            # s ([type]): Input features, i.e. side info

        Returns:
            [type]: [description]
        """
        c_i = self.ctx[i].calc(s)  # [n_samples, n_neurons]
        self.c[i] = c_i
        n_samples, n_neurons = c_i.shape[:2]
        # L_prev = self.L[i-1]
        logit_L_prev = self.logit_L[i-1]
        # Gated weights for each sample and each neuron
        # L_i = torch.empty((n_samples, n_neurons), device='cuda')
        logit_L_i = torch.empty((n_samples, n_neurons), device='cuda')

        for z in range(n_samples):  # ############# This is the slow part!
            w_i = self.gw[i]  # [n_neurons, n_contexts, layer_dim]
            W = torch.stack([w_i[j, c_i[z][j], :] for j in range(n_neurons)])
            # bias = math.e/(math.e + 1)
            logit_L_i[z, :] = W.matmul(logit_L_prev[z, :])
            # L_i[z, :] = torch.sigmoid(W.matmul(logit(L_prev[z, :])))
        # print('Max', torch.max(L_i), 'Min', torch.min(L_i), 'layer', i)
        # if torch.max(torch.sigmoid(logit_L_i)) > 0.99:  # TODO: This NaN still occurs
        #     raise Exception
        # return logit_L_i, L_i
        return logit_L_i

    def weights_init(self, K_prev, K_i):
        """Generate initial uniform weights for each context in layer's neurons

        Args:
            K_prev (Float): Size of previous layer
            K_i (Float): Size of current layer

        Returns:
            [Float] * [K_i, num_contexts, K_prev]: Weights
        """
        weights = torch.full((K_i, 2**self.n_context_fn, K_prev),
                              1.0/K_prev, device='cuda')
        return weights