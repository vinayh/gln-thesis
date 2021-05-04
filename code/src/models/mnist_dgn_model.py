from typing import Any, List

import torch

from src.models.mnist_gln_model import MNISTGLNModel
from src.models.modules.binary_dgn import BinaryDGN


class MNISTDGNModel(MNISTGLNModel):
    def __init__(*args, **kwargs):
        MNISTGLNModel.__init__(*args, **kwargs)
    
    def get_models(self, gpu=False):
        if self.hparams["gpu"]:
            return [BinaryDGN(hparams=self.hparams, binary_class=i).cuda()
                        for i in range(self.num_classes)]
        else:
            return [BinaryDGN(hparams=self.hparams, binary_class=i)
                        for i in range(self.num_classes)]
