import torch

from pytorch_lightning.metrics.classification import Accuracy
# from src.utils.helpers import to_one_vs_all
from typing import Any, List

import src.models.modules.binary_gln as BinaryGLN
BINARY_MODEL = BinaryGLN


class GLNModelNew(OVAModel):
    """
    LightningModule for classification (e.g. MNIST) using binary GLN with one-vs-all abstraction.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.t = 0
        self.save_hyperparameters()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.binary_criterion = torch.nn.BCEWithLogitsLoss()
        self.L1_loss = torch.nn.L1Loss(reduction="sum")
        self.params = self.get_model_params()
        # s_dim = hparams["input_size"]
        # self.l_sizes = (s_dim, hparams["lin1_size"],
        #                 hparams["lin2_size"], hparams["lin3_size"])
        # self.ctx_bias = True
        # self.w_clip = hparams["weight_clipping"]
        # s_dim = hparams["input_size"]
        # self.num_layers_used = hparams["num_layers_used"]
        # self.num_subctx = hparams["num_subcontexts"]
        # self.num_ctx = 2**self.num_subctx
        # self.X_all = X_all
        # self.y_all = y_all
        # self.binary_class = binary_class

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

    def training_step(self, batch: Any, batch_idx: int):
        loss = 0
        return {"loss": loss}

    def forward(self, batch: Any):
        return

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
