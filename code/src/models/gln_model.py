import torch

# from pytorch_lightning.metrics.classification import Accuracy
from typing import Any

from src.models.ova_model import OVAModel
from src.utils.helpers import to_one_vs_all
from src.utils.helpers import logit, nan_inf_in_tensor
import src.models.modules.rand_halfspace_gln as rand_hspace_gln


class GLNModel(OVAModel):
    """
    LightningModule for classification (e.g. MNIST) using binary GLN with one-vs-all abstraction.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t = 0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.binary_criterion = torch.nn.BCEWithLogitsLoss()
        self.params = self.get_model_params()
        self.register_buffer(
            "bmap",
            torch.tensor([2 ** i for i in range(self.hparams["num_subcontexts"])]),
        )

    def get_model_params(self):
        self.hparams.device = self.device
        # X_all, y_all_ova = self.get_plot_data()
        X_all, y_all_ova = None, None
        layer_sizes = self.layer_sizes_tuple(self.hparams)
        model_params = self.init_params(layer_sizes, X_all=X_all, y_all=y_all_ova)
        return model_params

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
        num_classes, layer_dim, _, input_dim = self.params["weights"][l_idx].shape
        if s.ndim == 1:
            s.unsqueeze(0)
        s = s[:, :-1]  # TODO: bias
        batch_size = s.shape[0]
        # c: [batch_size, input_dim]
        c = rand_hspace_gln.calc(
            s, self.params["ctx"][l_idx], self.bmap, self.hparams["gpu"]
        )
        # layer_bias = e / (e + 1)
        # TODO: bias
        # logit_x = torch.cat([logit_x, layer_bias * torch.ones_like(logit_x[:, :1])], dim=1)
        if nan_inf_in_tensor(logit_x):
            raise Exception
        # w_ctx: [batch_size, input_dim, output_layer_dim]
        w_ctx = torch.stack(
            [
                self.params["weights"][l_idx][range(layer_dim), c[j], :]
                for j in range(batch_size)
            ]
        )
        if nan_inf_in_tensor(w_ctx):
            raise Exception
        # logit_x_out: [batch_size, output_layer_dim]
        logit_x_out = torch.bmm(w_ctx, logit_x.unsqueeze(2)).flatten(start_dim=1)
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
            if y.ndim > 0:
                loss = torch.sigmoid(logit_x_out) - y.unsqueeze(1)
            else:
                loss = torch.sigmoid(logit_x_out) - y
            # w_delta: torch.einsum('ab,ac->acb', loss, logit_x)  # [batch_size, input_dim, output_layer_dim]
            w_delta = torch.bmm(loss.unsqueeze(2), logit_x.unsqueeze(1))
            w_new = torch.clamp(
                w_ctx - self.lr(self.hparams, self.t) * w_delta,
                min=-self.hparams["w_clip"],
                max=self.hparams["w_clip"],
            )
            if nan_inf_in_tensor(w_new):
                raise Exception
            # [batch_size, input_dim, output_layer_dim]
            with torch.no_grad():
                for j in range(batch_size):
                    self.params["weights"][l_idx][range(layer_dim), c[j], :] = w_new[j]
            #  TODO: Try adding layer_bias here to see if it helps
            if use_autograd:
                # w_ctx: [batch_size, input_dim, output_layer_dim]
                logit_x_out_updated = torch.bmm(w_new, logit_x.unsqueeze(2)).flatten(
                    start_dim=1
                )  # [batch_size, output_layer_dim]
                return logit_x_out, self.params, logit_x_out_updated
        return logit_x_out, self.params, None

    def base_layer(self, s_i):
        # TODO: Try using base_bias params to see if it helps
        p_clip = self.hparams["pred_clip"]
        logit_x_out = logit(torch.clamp(s_i, min=p_clip, max=1 - p_clip))
        if self.hparams["base_bias"]:
            if logit_x_out.ndim == 1:
                return torch.cat([logit_x_out[:-1], self.params["base_bias"]])
            else:
                return torch.cat([logit_x_out[:, :-1], self.params["base_bias"]])
        else:
            if logit_x_out.ndim == 1:
                return logit_x_out[:-1]
            else:
                return logit_x_out[:, :-1]

    def forward_helper(self, s_i, y_i, is_train):
        x_i = self.base_layer(s_i)
        # Gated layers
        for l_idx in range(self.hparams["num_layers_used"]):
            x_i, self.params, x_updated = self.gated_layer(
                x_i, s_i, y_i, l_idx, is_train=is_train
            )
            if is_train and self.hparams["train_autograd_params"]:
                # TODO: x_updated?
                self.autograd_fn(x_i, y_i, self.params["opt"][l_idx])
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

        x = torch.sigmoid(torch.tensor(x).type_as(s))

        logits = torch.stack(x).T.squeeze(0)
        loss = self.criterion(logits, y)
        acc = self.train_accuracy(torch.argmax(logits, dim=1), y)
        return loss, acc

    def init_params(self, layer_sizes, X_all=None, y_all=None):
        ctx, W, opt, biases = [], [], [], []
        base_bias = None
        num_contexts = 2 ** self.hparams["num_subcontexts"]
        use_autograd = self.hparams["train_autograd_params"]
        num_classes = self.hparams["num_classes"]
        p_clip = self.hparams["pred_clip"]
        # Base bias
        if self.hparams["base_bias"]:
            base_bias = torch.random.uniform(low=logit(p_clip), high=logit(1 - p_clip))
        # Params for gated layers
        for i in range(1, len(layer_sizes)):
            # input_dim, layer_dim = layer_sizes[i - 1] + 1, layer_sizes[i]
            input_dim, layer_dim = layer_sizes[i - 1], layer_sizes[i]
            layer_ctx = rand_hspace_gln.get_params(self.hparams, layer_dim)
            layer_W = (
                torch.ones(num_classes, layer_dim, num_contexts, input_dim) / input_dim
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
            ctx.append(ctx_param)
            W.append(W_param)
            opt.append(layer_opt)
            biases.append(bias_param)
        return {
            "ctx": ctx,
            "weights": W,
            "opt": opt,
            "biases": biases,
            "base_bias": base_bias,
        }

    @staticmethod
    def lr(hparams, t):
        if hparams["dynamic_lr"]:
            return min(hparams["lr"], hparams["lr"] / (1.0 + 1e-3 * t))
        else:
            return hparams["lr"]

    # For training in forward():
    @staticmethod
    def autograd_fn(h_updated, y_i, opt_i_layer):
        L1_loss_fn = torch.nn.L1Loss(reduction="sum")
        layer_logits_updated = torch.sigmoid(h_updated)
        loss = L1_loss_fn(layer_logits_updated.T, y_i)
        opt_i_layer.zero_grad()
        loss.backward()
        opt_i_layer.step()
