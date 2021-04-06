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
    # K = [2000, 2000, 1000, 500, 1]
    K = [4, 4, 4, 1]  # Num neurons in each layer including 0'th layer

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
                self.ctx[i] = HalfSpace(n_subcontexts=n_context_fn)
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
        self.logit_L[0] = logit(torch.rand(s.shape[0], self.K[0], device='cuda'))
        for i in range(1, len(self.K)):
            self.logit_L[i] = self.gated_layer(i, s)
        if torch.sigmoid(self.logit_L[-1]).isnan().any():
            raise Exception
        output = torch.sigmoid(self.logit_L[-1][:, 0])  # Undo logit for output
        # print('output', output)
        return output
    
    def update(self, eta, logit_L_prev_z, w_ijc, targets_z):
        delta = eta * (logit_geo_mix(logit_L_prev_z, w_ijc) - targets_z) * logit_L_prev_z
        new_w_ijc = torch.clamp(w_ijc - delta, min=-1.0, max=1.0)
        # if torch.max(new_w_ijc) > 0.99:
        #     raise Exception
        return new_w_ijc

    def update2(self, eta, logit_L_prev, w_ijc, targets):
        delta = eta * (logit_geo_mix(logit_L_prev, w_ijc) - targets)[:, None] * logit_L_prev
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
            for j in range(n_neurons):  # For each neuron
                c_ij = contexts[:, j]  # Contexts for neuron j for all samples
                # Weights for neuron j, all samples
                w_ijc = torch.index_select(self.gw[i][j], 0, c_ij.type(torch.cuda.LongTensor))
                # w_ic = torch.tensor([self.gw[i][j, c_ij[z]] for z in range(n_samples)], device='cuda')
                new_w_ijc = self.update2(eta, logit_L_prev, w_ijc, t)
                for z in range(n_samples):
                    self.gw[i][j, c_ij[z], :] = new_w_ijc[z]

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
        logit_L_prev = self.logit_L[i-1]
        # Gated weights for each sample and each neuron

        # OLD way, replaced by more vectorized method below
        # logit_L_i = torch.empty((n_samples, n_neurons), device='cuda')
        # w_i = self.gw[i]  # [n_neurons, n_contexts, layer_dim]
        # W_1 = None
        # for z in range(n_samples):  # ############# This is the slow part!
        #     W_1 = torch.stack([w_i[j, c_i[z][j], :] for j in range(n_neurons)])
        #     # bias = math.e/(math.e + 1)
        #     logit_L_i[z, :] = W_1.matmul(logit_L_prev[z, :])
        
        W = torch.empty((self.K[i], n_samples, self.K[i-1]), device='cuda')
        for j in range(n_neurons):
            W[j, :, :] = torch.index_select(self.gw[i][j], 0,
                                            c_i[:, j].type(torch.cuda.LongTensor))
        logit_L_i = torch.einsum('abc,bc->ba', W, logit_L_prev)
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