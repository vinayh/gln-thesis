import torch

from src.models.ova_model import OVAModel
from src.models.modules.binary_dgn import BinaryDGN
from src.utils.helpers import to_one_vs_all


class DGNModel(OVAModel):
    def __init__(*args, **kwargs):
        OVAModel.__init__(*args, **kwargs)

    def get_plot_data(self):
        if self.hparams["plot"]:
            datamodule = self.hparams["datamodule"]
            X_all, y_all = datamodule.get_all_data()
            y_all_ova = to_one_vs_all(y_all, self.num_classes)
        else:
            X_all = None
            y_all_ova = [0] * self.num_classes
        return X_all, y_all_ova

    def get_models(self, gpu=False):
        self.datamodule = self.hparams["datamodule"]
        X_all, y_all_ova = self.get_plot_data()
        if self.hparams["gpu"]:
            return [BinaryDGN(hparams=self.hparams, binary_class=i,
                              X_all=X_all,
                              y_all=y_all_ova[i]).cuda()
                    for i in range(self.num_classes)]
        else:
            return [BinaryDGN(hparams=self.hparams, binary_class=i,
                              X_all=X_all,
                              y_all=y_all_ova[i])
                    for i in range(self.num_classes)]

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        all_optimizers = []
        if self.hparams["train_context"]:
            # lr = 0.01
            def optim(p): return torch.optim.Adam(params=p, lr=1.0)

            for i in range(self.num_classes):  # For each binary model
                all_params_for_submodel = []
                for j in range(len(self.models[i].ctx)):  # For each subcontext
                    all_params_for_submodel.append(
                        self.models[i].ctx[j].hyperplanes)
                all_optimizers.append(optim(all_params_for_submodel))
        return all_optimizers
