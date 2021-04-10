from torch import nn


class HalfSpaceContext(nn.Module):
    def __init__(self, side_info_dim, layer_size):
        super().__init__()
        self.ctx_fn = nn.Linear(side_info_dim, layer_size)
        nn.init.normal_(self.ctx_fn.weight, mean=0.0, std=0.1)

    def calc_ctx(self, s):
        return (self.ctx_fn(s) > 0)

class GLNLayers(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.contexts = []
        self.weights = []
        l_sizes = (hparams["lin1_size"], hparams["lin2_size"], hparams["lin3_size"])
        self.weights.append(torch.full((l_sizes[i+1],
                                            2**hparams["num_subcontexts"],
                                            l_sizes[0]),
                                1.0/l_size))
        for i in range(1, len(l_sizes)):
            self.contexts.append(HalfSpaceContext(hparams["input_size"], l_sizes[i]))
            self.weights.append(torch.full((l_sizes[i],
                                            2**hparams["num_subcontexts"],
                                            l_sizes[i-1]),
                                1.0/l_sizes[i-1]))

                                # FIX EDGE CASES with l_size = 0 and last value

        self.model = nn.Linear(hparams["input_size"], hparams["lin1_size"])
    
    def get_contexts(self, hparams: dict):


    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)
