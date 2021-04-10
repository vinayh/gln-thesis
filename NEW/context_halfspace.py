import torch
import torch.nn as nn

class HalfSpace:
    def __init__(self, n_subcontexts=4):
        self.n_subcontexts = n_subcontexts
        self.ctx_fn = None

    def gen_layer(self, side_info_dim, layer_size):
        """Creates linear layer representing a context function
        for a provided previous and current layer size

        Args:
            n_subcontext (Float): Number of subcontext functions
            side_info_dim (Float): Size of side info
            curr_layer_dim (Float): Size of current layer

        Returns:
            # nn.Linear: Context function for specified layer sizes
            # Dimensionality is [n_layers, side_info_dim, curr_l_dim, n_subctx]
        """
        self.ctx_fn = []
        for _ in range(self.n_subcontexts):
            c = nn.Linear(side_info_dim, layer_size).to('cuda')
            nn.init.normal_(c.weight, mean=0.0, std=0.1) # TODO: check stdev in paper
            self.ctx_fn.append(c)

    def calc(self, s):
        """Calculates context indices for each input (side info) sample

        Args:
            s ([Float] * [batch_size,
                          side_info_dim]): Input features, i.e. side info

        Returns:
            [ctx_index * [n_samples, n_neurons]]: context ID in
                                                      0...2**n_subcontexts
        """
        n_samples = s.shape[0]
        n_neurons = self.ctx_fn[0].out_features  # Num of out_features (layer size)
        contexts = torch.zeros((n_samples, n_neurons), dtype=torch.int8,
                               device='cuda')
        for i in range(n_samples):  # For each input sample of side info
            for c in range(self.n_subcontexts):
                # Get subctx c result for all neurons in layer
                tmp = 2**c * ((self.ctx_fn[c](s[i, :])) > 0)
                contexts[i, :] += tmp[0, :]
        return contexts
