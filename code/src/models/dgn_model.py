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
