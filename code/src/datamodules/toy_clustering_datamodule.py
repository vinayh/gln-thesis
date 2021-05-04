from typing import Optional, Tuple

import torch
import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, Dataset, random_split
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import transforms

# from src.datamodules.mnist_utils import deskew_fn

class ToyClusteringDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        num_samples: int = 1500,
        train_val_test_split: Tuple[int, int, int] = (40, 5, 5),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        deskew: bool = False,
        fashionmnist: bool = False,
        **kwargs
    ):
        super().__init__()

        self.data_dir = data_dir
        self.num_samples = num_samples
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if fashionmnist:
            self.dataset_fn = FashionMNIST
        else:
            self.dataset_fn = MNIST

        if deskew:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Lambda(deskew_fn),
                 transforms.ToTensor()]
            )
        else:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))]
            )

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, 28, 28)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def gen_dataset(self, num_samples=50, num_classes=3, dim=2):
        centroids = 20 * np.random.rand(num_classes, dim) + (-10 * np.ones(dim))
        data = np.zeros((num_samples, dim+1))
        added = 0
        for i, c in enumerate(centroids):
            if i == len(centroids)-1:
                num_to_add = num_samples - added
            else:
                num_to_add = int(num_samples/num_classes)
            new_data = i * np.ones((num_to_add, dim+1))
            new_data[:, :dim] = np.random.multivariate_normal(mean=c,
                                                              cov=np.eye(dim),
                                                              size=num_to_add)
            data[added:added+num_to_add, :] = new_data
            added += num_to_add
        np.random.shuffle(data)
        data = torch.from_numpy(data)
        return data[:, :dim].float(), data[:, dim].long()

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        return

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        X, y = self.gen_dataset(self.num_samples, num_classes=3, dim=2)
        dataset = TensorDataset(X.float(), y.long())
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, self.train_val_test_split
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
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
