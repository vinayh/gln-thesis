import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy
from src.utils.helpers import to_one_vs_all
from typing import Any, List

import src.models.modules.binary_dgn as BinaryDGN
BINARY_MODEL = BinaryDGN


class DGNModelNew(LightningModule):
    """
    LightningModule for classification (e.g. MNIST) with one-vs-all abstraction.
    """

    def __init__(
        self,
        # input_size: int = 784,
        # lin1_size: int = 256,
        # lin2_size: int = 256,
        # lin3_size: int = 256,
        # num_classes: int = 10,
        # lr: float = 0.001,
        # weight_decay: float = 0.0005,
        **kwargs
    ):
        super().__init__()
        self.t = 0
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.num_classes = self.hparams["num_classes"]
        self.criterion = torch.nn.CrossEntropyLoss()
        self.binary_criterion = torch.nn.BCEWithLogitsLoss()
        self.L1_loss = torch.nn.L1Loss(reduction="sum")
        self.params = self.get_model_params()
        self.added_graph = False
        # Example input array for TensorBoard logger
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.hparams.device = self.device
        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }

    def get_plot_data(self):
        if self.hparams["plot"]:
            datamodule = self.hparams["datamodule"]
            X_all, y_all = datamodule.get_all_data()
            y_all_ova = to_one_vs_all(y_all, self.num_classes)
        else:
            X_all = None
            y_all_ova = [0] * self.num_classes
        return X_all, y_all_ova

    def get_model_params(self):
        self.datamodule = self.hparams["datamodule"]
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
            # Updating weights
            # if is_train:
            #     for i, p_i in enumerate(self.params):
            #         for opt_i in p_i["opt"]:
            #             opt_i.step()
            #             opt_i.zero_grad()

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
        # logits_binary, y_binary, loss = self.train_helper(
        #     x, y_ova, is_train=True)
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

    def test_helper(self, batch: Any):
        x, y = batch
        # if not self.added_graph:
        #     ex_inputs = (x, y, torch.tensor(False), torch.tensor(0))
        #     self.logger.experiment[0].add_graph(
        #         self, input_to_model=ex_inputs, verbose=False)
        #     self.added_graph = True
        y_ova = to_one_vs_all(y, self.num_classes, self.device)
        outputs = [BINARY_MODEL.forward(self.params[i], i, self.hparams,
                                        self.t, x, y_ova[i], is_train=False)
                   for i in range(self.num_classes)]
        logits = torch.stack(outputs).T.squeeze(0)
        loss = self.criterion(logits, y)
        logits_binary = torch.argmax(logits, dim=1)
        return loss, logits_binary, y

    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(
            self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(
            self.trainer.callback_metrics["train/loss"])
        self.log("train/acc_best",
                 max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best",
                 min(self.metric_hist["train/loss"]), prog_bar=False)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logits_binary, y_binary = self.test_helper(batch)
        acc = self.val_accuracy(logits_binary, y_binary)
        self.log("val/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False,
                 on_epoch=True, prog_bar=True)
        # return {"loss": loss, "logits_binary": logits_binary, "y_binary": y_binary}
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        self.metric_hist["val/acc"].append(
            self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(
            self.trainer.callback_metrics["val/loss"])
        self.log("val/acc_best",
                 max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best",
                 min(self.metric_hist["val/loss"]), prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        loss, logits_binary, y_binary = self.test_helper(batch)
        acc = self.test_accuracy(logits_binary, y_binary)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        # return {"loss": loss, "logits_binary": logits_binary, "y_binary": y_binary}
        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        # if self.hparams["plot"]:
        #     # for i in range(self.num_classes):
        #     for i in range(1):  # TODO: Currently only if class == 0
        #         self.params[i].plotter.save_animation('class_{}.gif'.format(i))
        pass

    def configure_optimizers(self):
        # optimizers = []
        # for p_i in self.params:
        #     # optimizers + BINARY_MODEL.configure_optimizers(p_i)
        #     optimizers = optimizers + p_i["opt"]
        # return optimizers
        pass
