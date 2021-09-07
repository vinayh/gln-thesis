from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy
from src.utils.helpers import to_one_vs_all

# from torchviz import make_dot


class OVAModel(LightningModule):
    """
    LightningModule for classification (e.g. MNIST) with one-vs-all abstraction.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.update(kwargs)  # TODO?
        self.automatic_optimization = False
        self.num_classes = self.hparams["num_classes"]
        self.models = self.get_models(self.hparams["gpu"])
        self.added_graph = False
        self.y_all_ova = [0] * self.num_classes
        # Example input array for TensorBoard logger
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.datamodule = self.hparams["datamodule"]
        self.t = 0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.binary_criterion = torch.nn.BCEWithLogitsLoss()
        self.hparams.device = self.device
        self.layer_sizes = self.layer_sizes_tuple(self.hparams)
        self.pretrain_complete = False
        self.X_all, self.y_all = None, None
        if (
            self.hparams["plot"]
            or self.hparams["ctx_evol_batch"]
            or self.hparams["ctx_evol_pretrain"]
            or self.hparams["ctx_svm_pretrain_force_redo"]
        ):
            self.X_all, self.y_all, self.y_all_ova = self.get_full_dataset()
        self.init_params()

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }

    @staticmethod
    def layer_sizes_tuple(hparams):
        if hparams["num_layers_used"] == 3:
            layer_sizes = (
                hparams["input_size"],
                hparams["lin1_size"],
                hparams["lin2_size"],
                hparams["lin3_size"],
            )
        elif hparams["num_layers_used"] == 4:
            layer_sizes = (
                hparams["input_size"],
                hparams["lin1_size"],
                hparams["lin2_size"],
                hparams["lin3_size"],
                hparams["lin4_size"],
            )
        else:
            raise Exception
        return layer_sizes

    # def get_example_input(self):
    #     ex_batch_size = 4
    #     ex_x = torch.rand(
    #         (ex_batch_size, self.hparams["input_size"]), device=self.device)
    #     ex_y = torch.zeros(self.hparams["num_classes"], device=self.device)
    #     ex_y[0] = 1
    #     ex_is_train = torch.tensor(True)
    #     return [ex_x, ex_y, ex_is_train]

    def get_models(self, gpu=False):
        """Implemented by child class: GLNModel, DGNModel, etc.

        Args:
            gpu (bool, optional): If models should be on GPU. Defaults to False.

        Returns:
            [BinaryGLN]: List of instances of a binary module (BinaryGLN, ...)
                         which will each be trained for a binary one-vs-all task
        """
        return NotImplementedError

    def training_step(self, batch: Any, batch_idx: int):
        self.t += 1
        self.hparams.device = self.device
        # assert(optimizer_idx is not None)
        loss, acc = self.forward(batch, is_train=True)
        # Log
        self.log("train/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "lr",
            self.lr(self.hparams, self.t),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(
            self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(
            self.trainer.callback_metrics["train/loss"]
        )
        self.log("train/acc_best",
                 max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best",
                 min(self.metric_hist["train/loss"]), prog_bar=False)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, acc = self.forward(batch, is_train=False)
        self.log("val/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
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
        loss, acc = self.forward(batch, is_train=False)
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

    def get_full_dataset(self):
        datamodule = self.hparams["datamodule"]
        X_all, y_all = datamodule.get_all_data()
        X_all = torch.cat([X_all, torch.ones_like(X_all[:, :1])], dim=1)
        y_all_ova = to_one_vs_all(y_all, self.num_classes)
        if self.hparams.gpu:
            return X_all.cuda(), y_all_ova.cuda()
        else:
            return X_all, y_all, y_all_ova

    def init_params(self, X_all=None, y_all=None):
        return NotImplementedError

    def base_layer(self, s_i):
        return NotImplementedError

    def gated_layer(
        self, logit_x, s, y, l_idx, is_train, use_autograd=False,
    ):
        return NotImplementedError

    def forward_helper(self, s_i, y_i, is_train):
        return NotImplementedError

    def forward(self, batch: Any, is_train=False):
        self.hparams.device = self.device
        x, y = batch
        y_ova = to_one_vs_all(y, self.num_classes, self.device).permute(1, 0)
        s = x.flatten(start_dim=1)
        s = torch.cat([s, torch.ones_like(s[:, :1])], dim=1)  # Add bias

        x = [
            self.forward_helper(s[i, :].unsqueeze(
                0), y_ova[i, :], is_train=is_train)
            for i in range(s.shape[0])
        ]

        logits = torch.sigmoid(torch.stack(x).squeeze(2).type_as(s))
        loss = self.criterion(logits, y)
        acc = self.train_accuracy(torch.argmax(logits, dim=1), y)
        return loss, acc

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
