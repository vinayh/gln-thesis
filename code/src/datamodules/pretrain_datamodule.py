"""
Needs to be used through child class that implements:
    - prepare_data()
    - setup()
    - get_pretrained()
    - self.dataset_name
"""

from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier

from os.path import join

import numpy as np

from src.utils.helpers import to_one_vs_all


class PretrainDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html

    ###

    The PretrainDataModule adds functions for generating/saving/loading
    SVM-pretrained context vectors, and can be used with a child
    class that also implements:
        - prepare_data()
        - setup()
        - get_pretrained()
    """

    def __init__(self, data_dir: str):
        super().__init__()

        self.data_dir = data_dir
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.X_all = None
        self.y_all = None

    def prepare_data(self):
        return

    def setup(self, stage: Optional[str] = None):
        return

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def get_all_data(self, include_test=False):
        self.setup()
        self.prepare_data()
        loader = DataLoader(
            dataset=self.data_train, batch_size=len(self.data_train), shuffle=False
        )
        X, y = next(iter(loader))
        return X.flatten(start_dim=1), y

    def get_svm_boundary(self, X, y):
        X, y = X.cpu().numpy(), y.cpu().numpy()
        svm = LinearSVC(dual=False)
        svm.fit(X, y)
        boundary = torch.from_numpy(np.append(svm.coef_, svm.intercept_))
        incorrect = torch.from_numpy(svm.predict(X) != y)
        return boundary, incorrect

    def get_pretrained_svm_helper(self, X_all, y_all_ova, num_classes):
        num_layers = 3
        pretrained = torch.zeros(num_classes, num_layers, X_all.shape[1] + 1)
        for i in range(num_classes):
            print("Pretraining for class: {}".format(i))
            # Start with all samples assumed to be incorrectly classified
            incorrect = torch.ones(X_all.shape[0], dtype=torch.uint8)
            num_incorrect = torch.sum(incorrect)
            X, y = X_all, y_all_ova[i]
            for k in range(num_layers):
                if num_incorrect == 0:  # If all correctly classified
                    break
                # Train on only previously incorrectly classified samples
                X, y = X[incorrect, :], y[incorrect]
                if len(torch.unique(y)) == 1:  # If all in same class
                    print(
                        "\tNot training layer: {} (remaining samples in same class), incorrect remaining: {}".format(
                            k, num_incorrect
                        )
                    )
                    break
                boundary, incorrect = self.get_svm_boundary(X, y)
                pretrained[i, k, :] = boundary
                num_incorrect = torch.sum(incorrect)
                print(
                    "\tTrained layer: {}, incorrect remaining: {}".format(
                        k, num_incorrect
                    )
                )
        return pretrained

    def get_pretrained_svm(
        self, X_all, y_all_ova, num_classes, model_name="", force_redo=False
    ):
        filepath = join(
            self.data_dir,
            "pretrained_{}_{}_coef.pt".format(self.dataset_name, model_name),
        )
        if force_redo:
            print("Training SVM models on dataset to generate SVM-based contexts")
            pretrained = self.get_pretrained_svm_helper(
                X_all, y_all_ova, num_classes)
            torch.save(pretrained, filepath)
        else:
            print("Loading previously saved SVM-based contexts")
            pretrained = torch.load(filepath)
        return pretrained

    ###
    # AdaBoost
    ###

    def adaboost_boundary(self, X, y):
        X, y = X.cpu().numpy(), y.cpu().numpy()
        svm = LinearSVC(dual=False)
        svm.fit(X, y)
        boundary = torch.from_numpy(np.append(svm.coef_, svm.intercept_))
        incorrect = torch.from_numpy(svm.predict(X) != y)
        return boundary, incorrect

    def pretrained_adaboost_helper_old(self, X_all, y_all_ova, num_classes):
        num_layers = 3
        pretrained = torch.zeros(num_classes, num_layers, X_all.shape[1] + 1)
        for i in range(num_classes):
            print("Pretraining for class: {}".format(i))
            # Start with all samples assumed to be incorrectly classified
            incorrect = torch.ones(X_all.shape[0], dtype=torch.uint8)
            num_incorrect = torch.sum(incorrect)
            X, y = X_all, y_all_ova[i]
            for k in range(num_layers):
                if num_incorrect == 0:  # If all correctly classified
                    break
                # Train on only previously incorrectly classified samples
                X, y = X[incorrect, :], y[incorrect]
                if len(torch.unique(y)) == 1:  # If all in same class
                    print(
                        "\tNot training layer: {} (remaining samples in same class), incorrect remaining: {}".format(
                            k, num_incorrect
                        )
                    )
                    break
                boundary, incorrect = self.get_svm_boundary(X, y)
                pretrained[i, k, :] = boundary
                num_incorrect = torch.sum(incorrect)
                print(
                    "\tTrained layer: {}, incorrect remaining: {}".format(
                        k, num_incorrect
                    )
                )
        return pretrained

    def pretrained_adaboost_helper(self, X_all, y_all_ova, num_classes):
        num_hyperplanes = 10
        pretrained = torch.zeros(num_classes, num_hyperplanes, X_all.shape[1])
        X = X_all.cpu()[:, :-1]  # Remove bias element
        y = y_all_ova.cpu()
        for i in range(num_classes):
            clf = AdaBoostClassifier(
                base_estimator=SGDClassifier(loss="log"), n_estimators=num_hyperplanes
            )
            clf.fit(X, y[i])
            for j in range(num_hyperplanes):
                pretrained[i, j, :-1] = torch.tensor(clf.estimators_[j].coef_)
                pretrained[i, j, -
                           1] = torch.tensor(clf.estimators_[j].intercept_)
        return pretrained

    def get_pretrained_adaboost(
        self, X_all, y_all_ova, num_classes, model_name="", force_redo=False
    ):
        filepath = join(
            self.data_dir,
            "pretrained_adaboost_{}_{}_coef.pt".format(
                self.dataset_name, model_name),
        )
        if force_redo:
<<<<<<< HEAD
            print("Training SVM models on dataset to generate AdaBoost-based contexts")
            pretrained = self.pretrained_adaboost_helper(
                X_all, y_all_ova, num_classes)
=======
            print("Training AdaBoost models on dataset to generate contexts")
            pretrained = self.pretrained_adaboost_helper(X_all, y_all_ova, num_classes)
>>>>>>> ef6b76acba5a30229f4512dfc38404ac01d38f1d
            torch.save(pretrained, filepath)
        else:
            print("Loading previously saved AdaBoost-based contexts")
            pretrained = torch.load(filepath)
        return pretrained
