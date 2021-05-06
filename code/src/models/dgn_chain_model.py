from typing import Any, List

import torch
import numpy as np

from pytorch_lightning import LightningModule
from sklearn.svm import SVC, LinearSVC
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, ConcatDataset

from src.models.gln_model import GLNModel
from src.models.modules.binary_dgn_chain import BinaryDGNChain


class DGNChainModel(GLNModel):
    """
    LightningModule for classification (e.g. MNIST) using chain of
    single-neuron binary DGN layers with context vectors pre-trained
    using SVM models, which are themselves trained on iteratively
    misclassified samples
    """
    def __init__(*args, **kwargs):
        GLNModel.__init__(*args, **kwargs)

    def get_dataset(self, with_test=False):
        data_dir = "../../../../data/"
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = MNIST(data_dir, train=True, download=True, transform=t)
        if with_test:
            testset = MNIST(data_dir, train=False, download=True, transform=t)
            dataset = ConcatDataset(datasets=[dataset, testset])
        loader = DataLoader(dataset=dataset, batch_size=len(dataset))
        X, y = next(iter(loader))
        return X.flatten(start_dim=1), y
    
    def get_svm_boundary(self, X, y):
        X, y = np.array(X), np.array(y)
        svm = LinearSVC(dual=False)
        svm.fit(X, y)
        boundary = torch.from_numpy(np.append(svm.coef_, svm.intercept_))
        incorrect = torch.from_numpy(svm.predict(X) != y)
        return boundary, incorrect
    
    def get_pretrained(self, X_all, y_all):
        y_all_ova = self.to_one_vs_all(y_all)
        num_layers = 3
        pretrained = torch.zeros(self.num_classes,
                                 num_layers,
                                 X_all.shape[1] + 1)
        for i in range(self.num_classes):
            print('Pretraining for class: {}'.format(i))
            # Start with all samples assumed to be incorrectly classified
            incorrect = torch.ones(X_all.shape[0], dtype=torch.uint8)
            X, y = X_all, y_all_ova[i]
            for k in range(num_layers):
                num_incorrect = torch.sum(incorrect)
                if num_incorrect == 0:  # If all correctly classified
                    break
                # Train on only previously incorrectly classified samples
                X, y = X[incorrect, :], y[incorrect]
                if len(torch.unique(y)) == 1:  # If all in same class
                    print('\tNot training layer: {} (remaining samples in same class), incorrect remaining: {}'
                    .format(k, num_incorrect))
                    break
                boundary, incorrect = self.get_svm_boundary(X, y)
                pretrained[i, k, :] = boundary
                print('\tTrained layer: {}, incorrect remaining: {}'
                    .format(k, num_incorrect))
        return pretrained
    
    def get_models(self, gpu=False):
        X, y = self.get_dataset(with_test=False)
        # self.pretrained = self.get_pretrained(X, y)
        # torch.save(self.pretrained, '../../../../pretrained_dgn_svm_coef.pt')
        self.pretrained = torch.load('../../../../pretrained_dgn_svm_coef.pt')
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
