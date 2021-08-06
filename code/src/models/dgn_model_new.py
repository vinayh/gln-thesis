import torch

# from pytorch_lightning.metrics.classification import Accuracy
from src.utils.helpers import to_one_vs_all
from typing import Any

import src.models.modules.binary_dgn as BinaryDGN
BINARY_MODEL = BinaryDGN


class DGNModelNew(OVAModel):
    """
    LightningModule for classification (e.g. MNIST) with one-vs-all abstraction.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.t = 0
        self.save_hyperparameters()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.binary_criterion = torch.nn.BCEWithLogitsLoss()
        self.L1_loss = torch.nn.L1Loss(reduction="sum")
        self.params = self.get_model_params()

    def get_model_params(self):
        X_all, y_all_ova = self.get_plot_data()
        num_neurons = self.num_neurons = (
            self.hparams["input_size"],
            self.hparams["lin1_size"],
            self.hparams["lin2_size"],
            self.hparams["lin3_size"])
        model_params = [BINARY_MODEL.init_params(num_neurons,
                                                 self.hparams,
                                                 binary_class=i,
                                                 X_all=X_all,
                                                 y_all=y_all_ova[i])
                        for i in range(self.num_classes)]
        return model_params

    def train_helper(self, x, y_ova, is_train=False):
        outputs = []
        for i, p_i in enumerate(self.params):  # For each binary model
            # Setup
            y_i = y_ova[i]
            if len(x.shape) > 1:
                s = x.flatten(start_dim=1)
            s_bias = torch.cat(
                [s, torch.ones(s.shape[0], 1, device=self.hparams.device)], dim=1)
            # Layers of network
            h = BINARY_MODEL.base_layer(s_bias)
            # For each layer, calculate loss of layer output, zero out grads
            # for layer weights, and perform update step using backward pass
            train_autograd_params = self.hparams["train_autograd_params"]
            for l_idx in range(self.hparams["num_layers_used"]):
                h, p_i, h_updated = BINARY_MODEL.gated_layer(p_i, self.hparams, h,
                                                             s_bias, y_i, l_idx,
                                                             is_train=is_train, is_gpu=False,
                                                             updated_outputs=train_autograd_params)
                layer_logits = torch.sigmoid(h)
                if is_train and train_autograd_params:
                    layer_logits_updated = torch.sigmoid(h_updated)
                    loss = self.L1_loss(layer_logits_updated.T, y_i)
                    p_i["opt"][l_idx].zero_grad()
                    # print(p_i["weights"][l_idx].grad)
                    loss.backward()
                    p_i["opt"][l_idx].step()
            # if is_train and self.hparams["plot"]:
            # if i == 1 and not (self.t % 5):
            #     print('\ntemp\n', params["ctx"][0])
            #     # TODO: Refactor plotting animations
            #     # def add_ctx_to_plot(xy, add_to_plot_fn):
            #     #     for l_idx in range(hparams["num_layers_used"]):
            #     #         Z = RandHalfSpaceDGN.calc_raw(
            #     #             xy, params["ctx"][l_idx])
            #     #         for b_idx in range(hparams["num_branches"]):
            #     #             add_to_plot_fn(Z[:, b_idx])
            #     # plotter.save_data(forward_fn, add_ctx_to_plot)
            outputs.append(layer_logits)
        logits = torch.stack(outputs).T.squeeze(0)
        return logits

    def training_step(self, batch: Any, batch_idx: int):
        # Train
        self.t += 1
        self.hparams.device = self.device
        # assert(optimizer_idx is not None)
        x, y = batch
        y_ova = to_one_vs_all(y, self.num_classes, self.device)
        logits = self.train_helper(x, y_ova, is_train=True)
        loss = self.criterion(logits, y.long())
        acc = self.train_accuracy(torch.argmax(logits, dim=1), y)
        # Log
        self.log("train/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("lr", BINARY_MODEL.lr(self.hparams),
                 on_step=True, on_epoch=True, prog_bar=True)

        # TODO: Use other optimizers besides idx 0
        # opt = self.optimizers()[0]
        # opt.zero_grad()
        # loss = self.compute_loss(batch)
        # self.manual_backward(loss)
        # opt.step()
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # return {"loss": loss, "logits_binary": logits_binary, "y_binary": y_binary}

        return {"loss": loss}

    def forward(self, batch: Any):
        x, y = batch
        # if not self.added_graph:
        #     ex_inputs = (x, y, torch.tensor(False), torch.tensor(0))
        #     self.logger.experiment[0].add_graph(
        #         self, input_to_model=ex_inputs, verbose=False)
        #     self.added_graph = True
        y_ova = to_one_vs_all(y, self.num_classes, self.device)
        outputs, plotter = [BINARY_MODEL.forward(self.params[i], self.hparams, i,
                                                 self.t, x, y_ova[i], is_train=False)
                            for i in range(self.num_classes)]
        logits = torch.stack(outputs).T.squeeze(0)
        loss = self.criterion(logits, y)
        logits_binary = torch.argmax(logits, dim=1)
        return loss, logits_binary, y
