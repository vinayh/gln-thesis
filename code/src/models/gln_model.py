from src.models.ova_model import OVAModel
from src.models.modules.binary_gln import BinaryGLN


class GLNModel(OVAModel):
    """
    LightningModule for classification (e.g. MNIST) using binary GLN with one-vs-all abstraction.
    """
    def __init__(*args, **kwargs):
        OVAModel.__init__(*args, **kwargs)

    def get_models(self, gpu=False):
        if self.hparams["gpu"]:
            return [BinaryGLN(hparams=self.hparams).cuda()
                    for i in range(self.num_classes)]
        else:
            return [BinaryGLN(hparams=self.hparams)
                    for i in range(self.num_classes)]
