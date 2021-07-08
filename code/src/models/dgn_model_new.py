from torch import optim
from torch.optim import optimizer
from src.utils.helpers import to_one_vs_all
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy

import src.models.modules.binary_dgn as BinaryDGN
MODEL_CLASS = BinaryDGN


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
        self.automatic_optimization = True
        self.save_hyperparameters()
        self.num_classes = self.hparams["num_classes"]
        self.criterion = torch.nn.CrossEntropyLoss()
        self.binary_criterion = torch.nn.BCEWithLogitsLoss()
        self.params = self.get_model_params()
        self.added_graph = False
        # Example input array for TensorBoard logger
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

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
        model_params = [MODEL_CLASS.init_params(num_neurons,
                                                self.hparams,
                                                binary_class=i,
                                                X_all=X_all,
                                                y_all=y_all_ova[i])
                        for i in range(self.num_classes)]
        if self.hparams["gpu"]:
            return [i.cuda() for i in model_params]
        else:
            return model_params

    def forward(self, x: torch.Tensor, y: torch.Tensor, is_train: bool,
                optimizer_idx: int):
        y_ova = to_one_vs_all(y, self.num_classes, self.device)
        self.hparams.device = self.device
        self.t += 1
        if optimizer_idx is not None:
            return MODEL_CLASS.forward(self.params[optimizer_idx],
                                       optimizer_idx, self.hparams,
                                       self.t, x, y_ova[optimizer_idx],
                                       is_train), y_ova[optimizer_idx]
        else:
            outputs = [MODEL_CLASS.forward(self.params[i], i, self.hparams,
                                           self.t, x, y_ova[i], is_train)
                       for i in range(self.num_classes)]
            return torch.stack(outputs).T.squeeze(0)

    def step(self, batch: Any, is_train=True, optimizer_idx=None):
        x, y = batch
        if optimizer_idx is not None:
            logits, y_ova = self.forward(x, y, is_train, optimizer_idx)
            loss = self.binary_criterion(logits.squeeze(1), y_ova.float())
            return loss, logits, y_ova
        else:
            logits = self.forward(x, y, is_train, None)
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            return loss, preds, y
        # if not self.added_graph:
        #     ex_inputs = (x, y, torch.tensor(False))
        #     self.logger.experiment[0].add_graph(
        #         self.params[0], input_to_model=ex_inputs, verbose=False)
        #     self.added_graph = True

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        print(optimizer_idx)
        # TODO: do something with optimizer_idx
        loss, preds, targets = self.step(
            batch, is_train=True, optimizer_idx=optimizer_idx)
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("lr", MODEL_CLASS.lr(self.hparams),
                 on_step=True, on_epoch=True, prog_bar=True)
        # TODO: Use other optimizers besides idx 0
        if optimizer_idx > 0:
            raise Exception
        # opt = self.optimizers()[0]
        # opt.zero_grad()
        # loss = self.compute_loss(batch)
        # self.manual_backward(loss)
        # opt.step()
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # return {"loss": loss, "preds": preds, "targets": targets}
        return {"loss": loss}

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
        loss, preds, targets = self.step(batch, is_train=False)
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False,
                 on_epoch=True, prog_bar=True)
        # return {"loss": loss, "preds": preds, "targets": targets}
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
        loss, preds, targets = self.step(batch, is_train=False)
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        # return {"loss": loss, "preds": preds, "targets": targets}
        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        # if self.hparams["plot"]:
        #     # for i in range(self.num_classes):
        #     for i in range(1):  # TODO: Currently only if class == 0
        #         self.params[i].plotter.save_animation('class_{}.gif'.format(i))
        pass

    def configure_optimizers(self):
        # all_optimizers = []
        # if self.hparams["train_context"]:
        #     # lr = 0.01
        #     def optim(p): return torch.optim.Adam(params=p, lr=1.0)

        #     for i in range(self.num_classes):  # For each binary model
        #         all_params_for_submodel = []
        #         for j in range(len(self.params[i].ctx)):  # For each subcontext
        #             all_params_for_submodel.append(
        #                 self.params[i].ctx[j].hyperplanes)
        #         all_optimizers.append(optim(all_params_for_submodel))
        #         print(list(self.params[i].parameters()))
        # return all_optimizers
        optimizers = []
        for p_i in self.params:
            optimizers.append(MODEL_CLASS.configure_optimizers(p_i))
        return optimizers
