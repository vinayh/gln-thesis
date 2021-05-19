from src.models.ova_model import OVAModel
from src.models.modules.binary_dgn_chain import BinaryDGNChain


class DGNChainModel(OVAModel):
    """
    LightningModule for classification (e.g. MNIST) using chain of
    single-neuron binary DGN layers with context vectors pre-trained
    using SVM models, which are themselves trained on iteratively
    misclassified samples
    """
    def __init__(*args, **kwargs):
        OVAModel.__init__(*args, **kwargs)

    def get_models(self, gpu=False):
        datamodule = self.hparams["datamodule"]
        self.pretrained = datamodule.get_pretrained(
            num_classes=self.num_classes,
            model_name='dgn_svm',
            use_saved_svm=self.hparams["use_saved_svm"])
        if self.hparams["gpu"]:
            return [BinaryDGNChain(hparams=self.hparams,
                                   binary_class=i,
                                   pretrained=self.pretrained[i]).cuda()
                    for i in range(self.num_classes)]
        else:
            return [BinaryDGNChain(hparams=self.hparams,
                                   binary_class=i,
                                   pretrained=self.pretrained[i])
                    for i in range(self.num_classes)]
