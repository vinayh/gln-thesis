import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy
from src.utils.helpers import to_one_vs_all

import src.models.modules.binary_gln as BinaryGLN
BINARY_MODEL = BinaryGLN


class GLNModelNew(LightningModule):
    """
    LightningModule for classification (e.g. MNIST) using binary GLN with one-vs-all abstraction.
    """

    def __init__(self, **kwargs):
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

    # def get_models(self, gpu=False):
    #     self.datamodule = self.hparams["datamodule"]
    #     if self.hparams["plot"]:
    #         datamodule = self.hparams["datamodule"]
    #         X_all, y_all = datamodule.get_all_data()
    #         y_all_ova = to_one_vs_all(y_all, self.num_classes)
    #     else:
    #         X_all = None
    #         y_all_ova = [0] * self.num_classes

    #     if self.hparams["gpu"]:
    #         return [BinaryGLN(hparams=self.hparams, binary_class=i,
    #                           X_all=X_all,
    #                           y_all=y_all_ova[i]).cuda()
    #                 for i in range(self.num_classes)]
    #     else:
    #         return [BinaryGLN(hparams=self.hparams, binary_class=i,
    #                           X_all=X_all,
    #                           y_all=y_all_ova[i])
    #                 for i in range(self.num_classes)]
