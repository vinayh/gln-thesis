from argparse import ArgumentError
import torch

# from pytorch_lightning.metrics.classification import Accuracy
from typing import Any

from torch import autograd

from src.models.ova_model import OVAModel
from src.utils.helpers import to_one_vs_all
from src.utils.helpers import logit, nan_inf_in_tensor
import src.models.modules.rand_halfspace_dgn_SINGLE as rand_hspace_dgn

L1_LOSS_FN = torch.nn.L1Loss(reduction="sum")


class DGNModel(OVAModel):
    """
    LightningModule for classification (e.g. MNIST) using DGN with one-vs-all abstraction.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gated_layer(self, h_in, s, y, l_idx, is_train):
        """Using provided input activations, context functions, and weights,
        returns the result of the DGN layer

        Args:
            h_in ([Float * [batch_size, input_dim]]): Input activations (with logit applied)
            s ([Float * [batch_size, s_dim]]): Batch of side info samples s
            y ([Float * [batch_size]]): Batch of binary targets
            l_idx (Int): Index of layer to use for selecting correct ctx and layer weights
            is_train (bool): Whether to train/update on this batch or not (for val/test)
        Returns:
            [Float * [batch_size, output_layer_dim]]: Output of DGN layer
        """
        # num_classes, _, output_dim, input_dim = self.W[l_idx].shape
        # c: [num_classes, num_branches, output_dim]
        c = rand_hspace_dgn.calc(s, self.ctx[l_idx])
        # layer_bias = e / (e + 1)  # TODO: bias
        if nan_inf_in_tensor(h_in):
            raise Exception
        # w_ctx: [num_classes, output_dim, input_dim]
        w_ctx = c.unsqueeze(2).matmul(self.W[l_idx]).squeeze(2)
        if nan_inf_in_tensor(w_ctx):
            raise Exception
        # h_out: [num_classes, output_dim]
        h_out = torch.bmm(w_ctx, h_in.unsqueeze(2)).squeeze(2)
        if nan_inf_in_tensor(h_out):
            raise Exception
        if is_train:
            r_out_unclipped = torch.sigmoid(h_out)
            r_out = torch.clamp(
                r_out_unclipped, min=logit(self.p_clip), max=logit(1 - self.p_clip),
            )
            # learn_gates: [num_classes, layer_dim]
            learn_gates = (
                torch.abs(r_out_unclipped - y.unsqueeze(1)) > self.p_clip
            ).float()
            w_grad = torch.bmm(
                ((r_out - y.unsqueeze(1)) * learn_gates).unsqueeze(2), h_in.unsqueeze(1)
            )
            # TODO: Try adding layer_bias here to see if it helps
            # w_delta: [num_classes, output_dim, num_branches, input_dim]
            w_delta = c.unsqueeze(3).matmul(w_grad.unsqueeze(2))
            if not self.autograd_weights:  # Manual weight update
                with torch.no_grad():
                    self.W[l_idx] = (
                        self.W[l_idx] - self.lr(self.hparams, self.t) * w_delta
                    )
            if self.use_autograd:
                loss = L1_LOSS_FN(r_out_unclipped, y.unsqueeze(1))
                self.opt[l_idx].zero_grad()
                loss.backward()
                if self.autograd_weights:
                    self.W[l_idx].grad = w_delta
                self.opt[l_idx].step()

        return h_out.detach(), None

    def base_layer(self, s_i):
        # TODO: Try using base_bias params to see if it helps
        logit_x_out = logit(torch.clamp(s_i, min=self.p_clip, max=1 - self.p_clip))
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
            # if is_train and self.hparams["train_autograd_params"]:
            #     # TODO: x_updated?
            #     self.autograd_fn(x_i, y_i, self.opt[l_idx])
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

    def init_params(self):
        self.use_autograd = self.hparams["autograd_local"]
        self.autograd_weights = self.hparams["autograd_local_w"]
        if self.autograd_weights and not self.use_autograd:
            raise ArgumentError
        self.ctx, self.W, self.opt, self.biases = [], [], [], []
        num_branches = self.hparams["num_branches"]
        num_classes = self.hparams["num_classes"]
        self.p_clip = self.hparams["pred_clip"]
        # Base bias
        base_bias = None
        if self.hparams["base_bias"]:
            base_bias = torch.random.uniform(
                low=logit(self.p_clip), high=logit(1 - self.p_clip)
            )
        # Params for gated layers
        for i in range(1, len(self.layer_sizes)):
            # input_dim, layer_dim = self.layer_sizes[i - 1] + 1, self.layer_sizes[i]
            input_dim, layer_dim = self.layer_sizes[i - 1], self.layer_sizes[i]
            layer_ctx = rand_hspace_dgn.get_params(self.hparams, layer_dim)
            layer_W = (
                torch.ones(num_classes, layer_dim, num_branches, input_dim) / input_dim
            )
            layer_bias = torch.empty(1, 1).uniform_(
                logit(self.p_clip), logit(1 - self.p_clip)
            )
            # TODO: Currently disabled grad to train on multiple GPUs without
            # error of autograd transferring data across devices
            if self.hparams["gpu"]:
                layer_ctx = layer_ctx.cuda()
                layer_W = layer_W.cuda()
                layer_bias = layer_bias.cuda()
            ctx_param = torch.nn.Parameter(layer_ctx, requires_grad=self.use_autograd)
            W_param = torch.nn.Parameter(layer_W, requires_grad=False)
            bias_param = torch.nn.Parameter(layer_bias, requires_grad=False)
            # ctx_param = torch.nn.Parameter(layer_ctx, requires_grad=self.use_autograd)
            if self.use_autograd and self.autograd_weights:
                layer_opt = torch.optim.SGD(
                    params=[ctx_param, W_param], lr=self.hparams["lr_autograd"]
                )
            elif self.use_autograd:
                layer_opt = torch.optim.SGD(
                    params=[ctx_param], lr=self.hparams["lr_autograd"]
                )
            else:
                layer_opt = None
            self.ctx.append(ctx_param)
            self.W.append(W_param)
            self.opt.append(layer_opt)
            self.biases.append(bias_param)
