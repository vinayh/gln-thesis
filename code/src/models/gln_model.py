import torch

# from pytorch_lightning.metrics.classification import Accuracy
from src.models.ova_model import OVAModel
from src.utils.helpers import to_one_vs_all
from typing import Any

import src.models.modules.binary_gln as BinaryGLN
BINARY_MODEL = BinaryGLN


class GLNModel(OVAModel):
    """
    LightningModule for classification (e.g. MNIST) using binary GLN with one-vs-all abstraction.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t = 0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.binary_criterion = torch.nn.BCEWithLogitsLoss()
        self.params = self.get_model_params()
        self.register_buffer("bmap", torch.tensor([2**i for i in range(self.hparams["num_subcontexts"])]))

    def get_model_params(self):
        self.hparams.device = self.device
        X_all, y_all_ova = self.get_plot_data()
        if self.hparams["num_layers_used"] == 4:
            num_neurons = self.num_neurons = (
                self.hparams["input_size"],
                self.hparams["lin1_size"],
                self.hparams["lin2_size"],
                self.hparams["lin3_size"],
                self.hparams["lin4_size"])
        else:
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

    @staticmethod
    def lr(hparams, t):
        return BINARY_MODEL.lr(hparams, t)

    # For training in BINARY_MODEL.forward():
    @staticmethod
    def autograd_fn(h_updated, y_i, opt_i_layer):
        L1_loss_fn = torch.nn.L1Loss(reduction="sum")
        layer_logits_updated = torch.sigmoid(h_updated)
        loss = L1_loss_fn(layer_logits_updated.T, y_i)
        opt_i_layer.zero_grad()
        loss.backward()
        opt_i_layer.step()

    def forward(self, batch: Any, is_train=False):
        self.hparams.device = self.device
        x, y = batch
        y_ova = to_one_vs_all(y, self.num_classes, self.device)
        use_autograd = self.hparams["train_autograd_params"]
        outputs = []
        for i, p_i in enumerate(self.params):  # For each binary model
            out_i = BINARY_MODEL.forward(p_i, self.hparams, i,
                                         self.t, x, y_ova[i],
                                         bmap=self.bmap,
                                         is_train=is_train,
                                         use_autograd=use_autograd,
                                         autograd_fn=self.autograd_fn)
            outputs.append(out_i)
        logits = torch.stack(outputs).T.squeeze(0)
        loss = self.criterion(logits, y)
        acc = self.train_accuracy(torch.argmax(logits, dim=1), y)
        return loss, acc

        # OLDER
        # if is_train and self.hparams["plot"]:
        # if i == 1 and not (self.t % 5):
        #     print('\ntemp\n', params["ctx"][0])
        #     # TODO: Refactor plotting animations
        #     # def add_ctx_to_plot(xy, add_to_plot_fn):
        #     #     for l_idx in range(hparams["num_layers_used"]):
        #     #         Z = rand_hspace_gln.calc_raw(
        #     #             xy, params["ctx"][l_idx])
        #     #         for b_idx in range(hparams["num_branches"]):
        #     #             add_to_plot_fn(Z[:, b_idx])
        #     # plotter.save_data(forward_fn, add_ctx_to_plot)
