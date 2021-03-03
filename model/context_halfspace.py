import torch
import torch.nn as nn

class HalfSpace:
    def __init__(self, n_subcontexts=4):
        self.n_subcontexts = n_subcontexts

    def init_fn(self, side_info_dim, layer_size):
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
        ctx_fn = []
        for _ in range(self.n_subcontexts):
            c = nn.Linear(side_info_dim, layer_size).to('cuda')
            nn.init.normal_(c.weight, mean=0.0, std=0.1) # TODO: check stdev in paper
            ctx_fn.append(c)
        return ctx_fn

    def calc(self, ctx_fn, s):
        """Calculates context indices for each input (side info) sample

        Args:
            ctx_fn ([nn.Linear] * [n_subcontext]): nn.Linear for each subcontext
            s ([Float] * [batch_size,
                          side_info_dim]): Input features, i.e. side info

        Returns:
            [ctx_index * [n_samples, n_neurons]]: context ID in
                                                      0...2**n_subcontexts
        """
        n_samples = s.shape[0]
        n_neurons = ctx_fn[0].out_features  # Num of out_features (layer size)
        contexts = torch.zeros((n_samples, n_neurons), dtype=torch.int8,
                               device='cuda')
        for i in range(n_samples):  # For each input sample of side info
            for c in range(self.n_subcontexts):
                # Get subctx c result for all neurons in layer
                tmp = 2**c * ((ctx_fn[c](s[i, :])) > 0)
                contexts[i, :] += tmp[0, :]
        return contexts

# OLD init_contexts
    # ctx_params = torch.normal(mean=0.0,
    #                           std=0.1,
    #                           size=(side_info_dim,
    #                                 curr_layer_dim,
    #                                 n_subcontext),
    #                           device='cuda')
    # torch.nn.init.normal_(ctx_params, mean=0.0, std=0.1)

# OLD calc_contexts
    # n_subcontexts = ctx_params.shape[2]
    # # ctx_out = torch.zeros((n_samples, n_neurons), dtype=torch.int8)
    # ctx_out = torch.zeros((n_samples, n_neurons),
    #                       dtype=torch.int8,
    #                       device='cuda')
    # for c in range(n_subcontexts):
    #     tmp = (s[:, 0, :].matmul(ctx_params[:, :, c]) > 0)
    #     assert(tmp.shape == ctx_out.shape)
    #     ctx_out += 2**c * tmp