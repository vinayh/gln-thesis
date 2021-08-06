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
        self.automatic_optimization = False
        self.num_classes = self.hparams["num_classes"]
        self.models = self.get_models(self.hparams["gpu"])
        self.added_graph = False
        self.hparams.device = self.device
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

    def forward(self, x: torch.Tensor, y: torch.Tensor, is_train: bool):
        """[summary]

        Args:
            x (torch.Tensor): [description]
            y (torch.Tensor): [description]
            is_train (bool): [description]

        Returns:
            [type]: [description]
        """
        # with torch.no_grad():
        y_ova = to_one_vs_all(y, self.num_classes, self.device)
        outputs = [self.models[i].forward(x, y_ova[i], is_train)
                   for i in range(self.num_classes)]
        return torch.stack(outputs).T.squeeze(0)

    # def step(self, batch: Any, is_train=True):
    #     """[summary]

    #     Args:
    #         batch (Any): [description]
    #         is_train (bool, optional): [description]. Defaults to True.

    #     Returns:
    #         [type]: [description]
    #     """
    #     # with torch.no_grad():
    #     x, y = batch
    #     logits = self.forward(x, y, is_train)
    #     loss = self.criterion(logits, y)
    #     preds = torch.argmax(logits, dim=1)
    #     if not self.added_graph:
    #         ex_inputs = (x, y, torch.tensor(False))
    #         self.logger.experiment[0].add_graph(
    #             self.models[0], input_to_model=ex_inputs, verbose=False)
    #         # make_dot(self.models[0](*ex_inputs)).render(
    #         #     "attached", format="png")
    #         self.added_graph = True
    #     return loss, preds, y

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
        loss, logits_binary, y_binary = self.forward(batch)
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
        loss, logits_binary, y_binary = self.forward(batch)
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

    def get_plot_data(self):
        if self.hparams["plot"]:
            datamodule = self.hparams["datamodule"]
            X_all, y_all = datamodule.get_all_data()
            y_all_ova = to_one_vs_all(y_all, self.num_classes)
        else:
            X_all = None
            y_all_ova = [0] * self.num_classes
        return X_all, y_all_ova
