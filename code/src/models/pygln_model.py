from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy
from src.utils.helpers import to_one_vs_all

# from src.ext.pygln import utils
from src.ext.pygln.pytorch import GLN

# from torchviz import make_dot


class PyGLNModel(LightningModule):
    """
    LightningModule for classification (e.g. MNIST) with one-vs-all abstraction.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.update(kwargs)  # TODO?
        self.automatic_optimization = False
        self.num_classes = self.hparams["num_classes"]
        self.model = self.get_models()
        self.added_graph = False
        # Example input array for TensorBoard logger
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.datamodule = self.hparams["datamodule"]

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }

    def get_models(self, gpu=False):
        """Implemented by child class: GLNModel, DGNModel, etc.

        Args:
            gpu (bool, optional): If models should be on GPU. Defaults to False.

        Returns:
            [BinaryGLN]: List of instances of a binary module (BinaryGLN, ...)
                         which will each be trained for a binary one-vs-all task
        """
        return GLN(
            layer_sizes=[
                self.hparams["lin1_size"],
                self.hparams["lin2_size"],
                self.hparams["lin3_size"],
            ],
            input_size=self.hparams["input_size"],
            num_classes=self.num_classes,
            pred_clipping=self.hparams["pred_clip"],
            learning_rate=self.hparams["lr"],
            bias=self.hparams["bias"],
            context_bias=self.hparams["ctx_bias"],
            context_map_size=self.hparams["num_subcontexts"],
        )

    def forward(self, batch: Any, is_train: bool):
        """[summary]

        Args:
            x (torch.Tensor): [description]
            y (torch.Tensor): [description]
            is_train (bool): [description]

        Returns:
            [type]: [description]
        """
        x, y = batch
        x = x.flatten(start_dim=1)
        if is_train:
            for n in range(x.shape[0]):
                self.model.predict(x[n : n + 1], target=y[n : n + 1])
            return torch.tensor(0.0), torch.tensor(0.0)
        else:
            preds = torch.tensor(
                [self.model.predict(x[n]) for n in range(x.shape[0])],
                device=self.device,
            )
            loss = torch.tensor(0.0)
            if preds.dim() > 1:
                preds = preds.squeeze()
            return loss, self.train_accuracy(preds, y)

    def training_step(self, batch: Any, batch_idx: int):
        # self.t += 1
        self.hparams.device = self.device
        # assert(optimizer_idx is not None)
        loss, acc = self.forward(batch, is_train=True)
        # Log
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log(
        #     "lr",
        #     self.lr(self.hparams, self.t),
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(
            self.trainer.callback_metrics["train/loss"]
        )
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, acc = self.forward(batch, is_train=False)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        # return {"loss": loss, "logits_binary": logits_binary, "y_binary": y_binary}
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        loss, acc = self.forward(batch, is_train=False)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        # return {"loss": loss, "logits_binary": logits_binary, "y_binary": y_binary}
        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        pass

    def get_plot_data(self):
        if self.hparams["plot"]:
            datamodule = self.hparams["datamodule"]
            X_all, y_all = datamodule.get_all_data()
            y_all_ova = to_one_vs_all(y_all, self.num_classes)
        else:
            X_all = None
            y_all_ova = [0] * self.num_classes
        return X_all, y_all_ova
