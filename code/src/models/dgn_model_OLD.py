import torch

# from pytorch_lightning.metrics.classification import Accuracy
from src.models.ova_model import OVAModel
from src.utils.helpers import to_one_vs_all
from typing import Any

import src.models.modules.binary_dgn_OLD as BinaryDGN

BINARY_MODEL = BinaryDGN


class DGNModel(OVAModel):
    """
    LightningModule for classification (e.g. MNIST) with one-vs-all abstraction.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t = 0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.binary_criterion = torch.nn.BCEWithLogitsLoss()
        self.L1_loss = torch.nn.L1Loss(reduction="sum")
        self.params = self.get_model_params()

    def get_model_params(self):
        self.hparams.device = self.device
        X_all, y_all_ova = self.get_plot_data()
        layer_sizes = self.layer_sizes_tuple(self.hparams)
        model_params = [
            BINARY_MODEL.init_params(
                layer_sizes,
                self.hparams,
                binary_class=i,
                X_all=X_all,
                y_all=y_all_ova[i],
            )
            for i in range(self.num_classes)
        ]
        return model_params

    @staticmethod
    def lr(hparams, t):
        return BINARY_MODEL.lr(hparams, t)

    # For training in BINARY_MODEL.forward():
    @staticmethod
    def autograd_fn(r_updated, y_i, opt_i_layer):
        """Function to pass to BINARY_MODEL.forward() as callback
        for training autograd params after each gated_layer

        Args:
            r_updated ([type]): Layer outputs (with sigmoid applied)
            y_i ([type]): [description]
            opt_i_layer ([type]): [description]
        """
        L1_loss_fn = torch.nn.L1Loss(reduction="sum")
        loss = L1_loss_fn(r_updated.T, y_i)
        opt_i_layer.zero_grad()
        loss.backward()
        opt_i_layer.step()

    def forward(self, batch: Any, is_train=False):
        """Runs full one-vs-all forward pass
        Training model parameters if is_train=True, including
        autograd if hparams["train_autograd_params] is true

        Args:
            batch (Any): [description]
            is_train (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        self.hparams.device = self.device
        x, y = batch
        y_ova = to_one_vs_all(y, self.num_classes, self.device)
        outputs = []
        for i, p_i in enumerate(self.params):  # For each binary model
            out_i = BINARY_MODEL.forward(
                p_i,
                self.hparams,
                i,
                self.t,
                x,
                y_ova[i],
                is_train=is_train,
                autograd_fn=self.autograd_fn,
            )
            outputs.append(out_i)
        logits = torch.stack(outputs).T.squeeze(0)
        loss = self.criterion(logits, y)
        acc = self.train_accuracy(torch.argmax(logits, dim=1), y)
        return loss, acc

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
        # outputs.append(layer_logits)
