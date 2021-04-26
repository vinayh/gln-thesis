from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy

from src.models.modules.binary_gln import BinaryGLN


class MNISTGLNModel(LightningModule):
    """
    LightningModule for MNIST classification using binary GLN with one-vs-all abstraction.
    """

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        num_classes: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        **kwargs
    ):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters()
        self.num_classes = self.hparams["num_classes"]
        self.criterion = torch.nn.NLLLoss()

        if self.hparams["gpu"]:
            self.models = [BinaryGLN(hparams=self.hparams).cuda()
                        for i in range(self.num_classes)]
        else:
            self.models = [BinaryGLN(hparams=self.hparams)
                        for i in range(self.num_classes)]

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }
    
    def to_one_vs_all(self, targets):
        """
        Input: Torch tensor of target values (categorical labels)
        Returns: List of Torch tensors containing one-hot targets
                    for each class (used for one-vs-all models)
        """
        ova_targets = torch.zeros((self.num_classes, len(targets)),
                                dtype=torch.int, requires_grad=False,
                                device=self.device)
        for i in range(self.num_classes):
            ova_targets[i, :][targets == i] = 1
        return ova_targets

    def forward(self, x: torch.Tensor, y: torch.Tensor, is_train: bool):
        with torch.no_grad():
            y_ova = self.to_one_vs_all(y)
            outputs = [self.models[i](x, y_ova[i], is_train).squeeze()
                       for i in range(self.num_classes)]
        return torch.stack(outputs).T

    def step(self, batch: Any, is_train=True):
        with torch.no_grad():
            x, y = batch
            logits = self.forward(x, y, is_train)
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        with torch.no_grad():
            loss, preds, targets = self.step(batch, is_train=True)
            # log train metrics
            acc = self.train_accuracy(preds, targets)
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            # Log learning rate in progress bar
            self.log("lr", self.models[0].lr, on_step=True, on_epoch=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(self.trainer.callback_metrics["train/loss"])
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    def validation_step(self, batch: Any, batch_idx: int):
        with torch.no_grad():
            loss, preds, targets = self.step(batch, is_train=False)
            # log val metrics
            acc = self.val_accuracy(preds, targets)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # log best so far val acc and val loss
        self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        with torch.no_grad():
            loss, preds, targets = self.step(batch, is_train=False)
            # log test metrics
            acc = self.test_accuracy(preds, targets)
            self.log("test/loss", loss, on_step=False, on_epoch=True)
            self.log("test/acc", acc, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        pass
