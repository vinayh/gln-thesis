import torch

# from pytorch_lightning.metrics.classification import Accuracy
from typing import Any

from src.models.ova_model import OVAModel
from src.utils.helpers import to_one_vs_all
from src.utils.helpers import logit, nan_inf_in_tensor
import src.models.modules.rand_halfspace_gln as rand_hspace_gln


class GLNModel(OVAModel):
    """
    LightningModule for classification (e.g. MNIST) using GLN with one-vs-all abstraction.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gated_layer(
        self, logit_x, s, y, l_idx, is_train, use_autograd=False,
    ):
        """Using provided input activations, context functions, and weights,
        returns the result of the GLN layer

        Args:
            logit_x ([Float * [batch_size, input_dim]]): Input activations (with logit applied)
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
            y ([Float * [batch_size]]): Batch of binary targets
            l_idx (Int): Index of layer to use for selecting correct ctx and layer weights
            is_train (bool): Whether to train/update on this batch or not (for val/test)
        Returns:
            [Float * [batch_size, output_layer_dim]]: Output of GLN layer
        """
        num_classes, _, output_dim, input_dim = self.W[l_idx].shape
        # c: [num_classes, output_dim]
        c = rand_hspace_gln.calc(s, self.ctx[l_idx], self.bmap, self.hparams["gpu"])
        # layer_bias = e / (e + 1)  # TODO: bias
        if nan_inf_in_tensor(logit_x):
            raise Exception
        # w_ctx: [num_classes, output_dim, input_dim]
        w_ctx = self.W[l_idx][
            torch.arange(num_classes).reshape(-1, 1),
            c,
            torch.arange(output_dim).reshape(1, -1),
            :,
        ]
        if nan_inf_in_tensor(w_ctx):
            raise Exception
        # logit_x_out: [num_classes, output_dim]
        # logit_x_out = torch.matmul(w_ctx, logit_x)
        logit_x_out = torch.bmm(w_ctx, logit_x.unsqueeze(2)).squeeze(2)
        # Clamp to pred_clip
        logit_x_out = torch.clamp(
            logit_x_out,
            min=logit(self.hparams["pred_clip"]),
            max=logit(1 - self.hparams["pred_clip"]),
        )
        if nan_inf_in_tensor(logit_x_out):
            raise Exception
        if is_train:
            # loss: [batch_size, output_layer_dim]
            loss = torch.sigmoid(logit_x_out) - y.unsqueeze(1)
            # w_delta: [num_classes, output_dim, input_dim]
            # w_delta: torch.einsum('ab,ac->acb', loss, logit_x)
            w_delta = torch.bmm(loss.unsqueeze(2), logit_x.unsqueeze(1))
            w_new = torch.clamp(
                w_ctx - self.lr(self.hparams, self.t) * w_delta,
                min=-self.hparams["w_clip"],
                max=self.hparams["w_clip"],
            )
            if nan_inf_in_tensor(w_new):
                raise Exception
            # [num_classes, output_dim, input_dim]
            with torch.no_grad():
                self.W[l_idx][
                    torch.arange(num_classes).reshape(-1, 1),
                    c,
                    torch.arange(output_dim).reshape(1, -1),
                    :,
                ] = w_new
                # for i in range(self.hparams["num_classes"]):
                #     self.W[l_idx][
                #         i, c[i, range(c.shape[1])], range(c.shape[1]), :
                #     ] = w_new[i]
            #  TODO: Try adding layer_bias here to see if it helps

            # TODO: Fix autograd with GLN rewrite
            if use_autograd:
                # w_ctx: [num_classes, output_dim, input_dim]
                logit_x_out_updated = torch.bmm(w_new, logit_x.unsqueeze(2)).flatten(
                    start_dim=1
                )  # [num_classes, output_layer_dim]
                return logit_x_out, logit_x_out_updated
        return logit_x_out, None

    def base_layer(self, s_i):
        # TODO: Try using base_bias params to see if it helps
        p_clip = self.hparams["pred_clip"]
        logit_x_out = logit(torch.clamp(s_i, min=p_clip, max=1 - p_clip))
        logit_x_out = logit_x_out.expand(self.hparams["num_classes"], -1)
        if self.hparams["base_bias"]:
            return torch.cat([logit_x_out[:, :-1], self.biases])
        else:
            return logit_x_out[:, :-1]

    def forward_helper(self, s_i, y_i, is_train):
        s_i = s_i.squeeze(0)
        x_i = self.base_layer(s_i)
        # Gated layers
        for l_idx in range(self.hparams["num_layers_used"]):
            x_i, x_updated = self.gated_layer(x_i, s_i, y_i, l_idx, is_train=is_train)
            if is_train and self.hparams["train_autograd_params"]:
                # TODO: x_updated?
                self.autograd_fn(x_i, y_i, self.opt[l_idx])
        return x_i

    def forward(self, batch: Any, is_train=False):
        self.hparams.device = self.device
        x, y = batch
        y_ova = to_one_vs_all(y, self.num_classes, self.device).permute(1, 0)
        s = x.flatten(start_dim=1)
        s = torch.cat([s, torch.ones_like(s[:, :1])], dim=1)  # Add bias

        x = [
            self.forward_helper(s[i, :].unsqueeze(0), y_ova[i, :], is_train=is_train)
            for i in range(s.shape[0])
        ]

        logits = torch.sigmoid(torch.stack(x).squeeze(2).type_as(s))
        loss = self.criterion(logits, y)
        acc = self.train_accuracy(torch.argmax(logits, dim=1), y)
        return loss, acc

    def init_params(self, X_all=None, y_all=None):
        self.ctx, self.W, self.opt, self.biases = [], [], [], []
        base_bias = None
        num_contexts = 2 ** self.hparams["num_subcontexts"]
        use_autograd = self.hparams["train_autograd_params"]
        num_classes = self.hparams["num_classes"]
        p_clip = self.hparams["pred_clip"]
        # Base bias
        if self.hparams["base_bias"]:
            base_bias = torch.random.uniform(low=logit(p_clip), high=logit(1 - p_clip))
        bmap = torch.tensor([2 ** i for i in range(self.hparams["num_subcontexts"])])
        # Params for gated layers
        for i in range(1, len(self.layer_sizes)):
            # input_dim, layer_dim = self.layer_sizes[i - 1] + 1, self.layer_sizes[i]
            input_dim, layer_dim = self.layer_sizes[i - 1], self.layer_sizes[i]
            layer_ctx = rand_hspace_gln.get_params(self.hparams, layer_dim)
            layer_W = (
                torch.ones(num_classes, num_contexts, layer_dim, input_dim) / input_dim
            )
            layer_bias = torch.empty(1, 1).uniform_(logit(p_clip), logit(1 - p_clip))
            # TODO: Currently disabled grad to train on multiple GPUs without
            # error of autograd transferring data across devices
            ctx_param = torch.nn.Parameter(layer_ctx, requires_grad=False)
            W_param = torch.nn.Parameter(layer_W, requires_grad=False)
            bias_param = torch.nn.Parameter(layer_bias, requires_grad=False)
            if self.hparams["gpu"]:
                ctx_param = ctx_param.cuda()
                W_param = W_param.cuda()
                bias_param = bias_param.cuda()
            # ctx_param = torch.nn.Parameter(layer_ctx, requires_grad=use_autograd)
            layer_opt = (
                torch.optim.SGD(params=[layer_ctx], lr=0.1) if use_autograd else None
            )
            self.ctx.append(ctx_param)
            self.W.append(W_param)
            self.opt.append(layer_opt)
            self.biases.append(bias_param)
            self.bmap = bmap
