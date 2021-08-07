import torch

# from pytorch_lightning.metrics.classification import Accuracy
from src.models.ova_model import OVAModel
from src.utils.helpers import to_one_vs_all
from typing import Any

import src.models.modules.binary_gln as BinaryGLN


class GLNModelNew(OVAModel):
    """
    LightningModule for classification (e.g. MNIST) using binary GLN with one-vs-all abstraction.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t = 0
        self.BINARY_MODEL = BinaryGLN
        self.save_hyperparameters()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.binary_criterion = torch.nn.BCEWithLogitsLoss()
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
        model_params = [self.BINARY_MODEL.init_params(num_neurons,
                                                      self.hparams,
                                                      binary_class=i,
                                                      X_all=X_all,
                                                      y_all=y_all_ova[i])
                        for i in range(self.num_classes)]
        return model_params

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
        x, y = batch
        y_ova = to_one_vs_all(y, self.num_classes, self.device)
        use_autograd = self.hparams["train_autograd_params"]
        outputs = []
        for i, p_i in enumerate(self.params):  # For each binary model
            out_i = self.BINARY_MODEL.forward(p_i, self.hparams, i,
                                              self.t, x, y_ova[i],
                                              is_train=is_train,
                                              use_autograd=use_autograd,
                                              autograd_fn=self.autograd_fn)
            outputs.append(out_i)
        logits = torch.stack(outputs).T.squeeze(0)
        loss = self.criterion(logits, y)
        acc = self.train_accuracy(torch.argmax(logits, dim=1), y)
        return loss, acc

        # OLD
        # if len(x.shape) > 1:
        #     s = x.flatten(start_dim=1)
        # s_bias = torch.cat(
        #     [s, torch.ones(s.shape[0], 1, device=self.hparams.device)], dim=1)
        # # Layers of network
        # h = BINARY_MODEL.base_layer(s_bias, self.hparams["input_size"])
        # # For each layer, calculate loss of layer output, zero out grads
        # # for layer weights, and perform update step using backward pass
        # train_autograd_params = self.hparams["train_autograd_params"]
        # for l_idx in range(self.hparams["num_layers_used"]):
        #     h, p_i, h_updated = BINARY_MODEL.gated_layer(p_i, self.hparams, h,
        #                                                  s_bias, y_i, l_idx, self.t,
        #                                                  is_train=is_train, is_gpu=False,
        #                                                  updated_outputs=train_autograd_params)
        #     layer_logits = torch.sigmoid(h)
        #     if is_train and train_autograd_params:
        #         layer_logits_updated = torch.sigmoid(h_updated)
        #         loss = self.L1_loss(layer_logits_updated.T, y_i)
        #         p_i["opt"][l_idx].zero_grad()
        #         # print(p_i["weights"][l_idx].grad)
        #         loss.backward()
        #         p_i["opt"][l_idx].step()

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

    # def forward_old(self, batch: Any, is_train=False):
    #     x, y = batch
    #     y_ova = to_one_vs_all(y, self.num_classes, self.device)
    #     self.hparams.device = self.device
    #     outputs = [BINARY_MODEL.forward(self.params[i], self.hparams, i,
    #                                     self.t, x, y_ova[i], is_train=is_train)
    #                for i in range(self.num_classes)]
    #     logits = torch.stack(outputs).T.squeeze(0)
    #     loss = self.criterion(logits, y)
    #     # logits_binary = torch.argmax(logits, dim=1)
    #     acc = self.train_accuracy(torch.argmax(logits, dim=1), y)
    #     return loss, acc
    #     # if not self.added_graph:
    #     #     ex_inputs = (x, y, torch.tensor(False), torch.tensor(0))
    #     #     self.logger.experiment[0].add_graph(
    #     #         self, input_to_model=ex_inputs, verbose=False)
    #     #     self.added_graph = True
