import torch

# from pytorch_lightning.metrics.classification import Accuracy
from typing import Any

from src.models.ova_model import OVAModel
from src.utils.helpers import to_one_vs_all
from src.utils.helpers import logit, nan_inf_in_tensor, entropy

# from src.models.modules.pretrain import pretrain_svm_helper
import src.models.modules.rand_halfspace_gln as rand_hspace_gln


class GLNModel(OVAModel):
    """
    LightningModule for classification (e.g. MNIST) using GLN with one-vs-all abstraction.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gated_layer(
        self, logit_x, s, y, l_idx, is_train,
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
        logit_x_out = torch.bmm(w_ctx, logit_x.unsqueeze(2)).squeeze(2)
        # Clamp to pred_clip
        logit_x_out = torch.clamp(
            logit_x_out, min=logit(self.p_clip), max=logit(1 - self.p_clip),
        )
        if nan_inf_in_tensor(logit_x_out):
            raise Exception
        if is_train:
            # loss: [batch_size, output_layer_dim]
            loss = torch.sigmoid(logit_x_out) - y.unsqueeze(1)
            # w_delta: [num_classes, output_dim, input_dim]
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
            #  TODO: Try adding layer_bias here to see if it helps
        return logit_x_out, None

    def base_layer(self, s_i):
        # TODO: Try using base_bias params to see if it helps
        logit_x_out = logit(torch.clamp(s_i, min=self.p_clip, max=1 - self.p_clip))
        logit_x_out = logit_x_out.expand(self.num_classes, -1)
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
            if is_train and self.hparams["train_evol_sample"]:
                raise NotImplementedError
        return x_i

    def forward(self, batch: Any, is_train=False):
        # Pretraining
        if not self.pretrain_complete:
            self.pretrain()

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

    def pretrain(self):
        if self.hparams["ctx_evol_pretrain"]:
            for l_idx in range(len(self.ctx)):
                self.ctx[l_idx] = self.gln_evol_pretrain(self.ctx[l_idx])
        elif self.hparams["ctx_svm_pretrain"]:
            for l_idx in range(len(self.ctx)):
                self.ctx[l_idx] = self.gln_svm_pretrain(self.ctx[l_idx])
        self.pretrain_complete = True

    def init_params(self):
        self.ctx, self.W, self.opt, self.biases = [], [], [], []
        self.num_contexts = 2 ** self.hparams["num_subcontexts"]
        # use_autograd = self.hparams["train_autograd_params"]
        num_classes = self.num_classes
        self.p_clip = self.hparams["pred_clip"]
        # Base bias
        base_bias = None
        if self.hparams["base_bias"]:
            base_bias = torch.random.uniform(
                low=logit(self.p_clip), high=logit(1 - self.p_clip)
            )
        bmap = torch.tensor([2 ** i for i in range(self.hparams["num_subcontexts"])])
        # Params for gated layers
        for i in range(1, len(self.layer_sizes)):
            # input_dim, layer_dim = self.layer_sizes[i - 1] + 1, self.layer_sizes[i]
            input_dim, layer_dim = self.layer_sizes[i - 1], self.layer_sizes[i]
            layer_ctx = rand_hspace_gln.get_params(self.hparams, layer_dim)
            layer_W = (
                torch.ones(num_classes, self.num_contexts, layer_dim, input_dim)
                / input_dim
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
                bmap = bmap.cuda()
            ctx_param = torch.nn.Parameter(layer_ctx, requires_grad=False)
            W_param = torch.nn.Parameter(layer_W, requires_grad=False)
            bias_param = torch.nn.Parameter(layer_bias, requires_grad=False)
            # ctx_param = torch.nn.Parameter(layer_ctx, requires_grad=use_autograd)
            # layer_opt = (
            #     torch.optim.SGD(params=[layer_ctx], lr=0.1) if use_autograd else None
            # )
            self.ctx.append(ctx_param)
            self.W.append(W_param)
            # self.opt.append(layer_opt)
            self.biases.append(bias_param)
            self.bmap = bmap

    ###
    # Pretraining methods
    ###

    def gln_svm_pretrain(self, ctx):
        pretrained = self.datamodule.get_pretrained(
            self.X_all,
            self.y_all_ova,
            self.hparams.num_classes,
            model_name="GLN",
            force_redo=self.hparams.ctx_svm_pretrain_force_redo,
        )

    def gln_evol_fitness(self, ctx):
        output_dim = ctx.shape[2]
        # c: [num_samples, num_classes, output_dim]
        c = torch.stack(
            [
                rand_hspace_gln.calc(X_i, ctx, self.bmap, self.hparams["gpu"])
                for X_i in self.X_all
            ]
        )
        # Calculate initial entropy without context functions
        num_total = self.y_all_ova.shape[1]
        entropies = torch.zeros(self.num_classes).type_as(ctx)
        for i in range(self.num_classes):
            num_true = torch.sum(self.y_all_ova[i])
            entropies[i] = entropy(num_true / num_total)

        fitness = torch.zeros(output_dim, self.num_classes).type_as(ctx)
        for k in range(output_dim):  # For each neuron/output_dim
            # print("ctx_evol_pretrain: - Neuron {}".format(k))
            for i in range(self.num_classes):  # For each class
                fitness[k, i] = entropies[i]  # Set to init entropy without ctx fn
                y_all = self.y_all_ova[i, :]
                for j in range(self.num_contexts):
                    idx = (c[:, i, k] == torch.tensor(j)).nonzero().flatten()
                    num_in_ctx = len(idx)
                    p_true = torch.sum(y_all[idx]) / num_in_ctx
                    fitness[k, i] -= entropy(p_true)
                    if fitness[k, i].isnan():
                        raise Exception
        return fitness  # Sum over classes to get fitness of neurons

    def gln_evol_pretrain(self, ctx):
        assert self.X_all is not None
        N = 10  # Number of iterations
        n = 10  # Number of episodes
        lr = 0.01  # Learning rate
        sigma = 0.1  # Noise std
        output_dim = ctx.shape[2]
        for t in range(N):
            print("ctx_evol_pretrain: Evol. iteration {}".format(t))
            eps = torch.zeros(n, *ctx.shape).type_as(ctx).normal_(mean=0, std=1)
            F = torch.zeros(n, output_dim, self.num_classes).type_as(ctx)
            for num_ep in range(n):
                print("ctx_evol_pretrain: - Episode {}".format(num_ep))
                F[num_ep] = self.gln_evol_fitness(ctx + sigma * eps[num_ep])
                # delta = torch.sum(torch.bmm(eps, F), dim=1)
                for k in range(output_dim):
                    for i in range(self.num_classes):
                        ctx[i, :, k, :] = (
                            ctx[i, :, k, :]
                            + (lr / (n * sigma))
                            * F[num_ep, k, i]
                            * eps[num_ep, i, :, k, :]
                        )
        # TODO: Check and fix above, try implementing evol method in batch
        return ctx
