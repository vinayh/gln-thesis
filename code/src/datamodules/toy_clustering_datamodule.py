from typing import Optional, Tuple

import torch
import numpy as np

from torch.utils.data import TensorDataset, random_split

from src.datamodules.pretrain_datamodule import PretrainDataModule


class ToyClusteringDataModule(PretrainDataModule):
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
        super().__init__(data_dir=data_dir)

        self.dataset_name = "toyclustering"
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def gen_dataset(self, num_samples=50, num_classes=3, dim=2):
        np.random.seed(42)
        centroids = np.random.rand(num_classes, dim)
        centroids[0] = np.mean(centroids[1:], axis=0)
        # Scale and center centroids
        centroids = 20 * (centroids - np.mean(centroids, axis=0))
        data = np.zeros((num_samples, dim + 1))
        added = 0
        for i, c in enumerate(centroids):
            if i == len(centroids) - 1:  # If last centroid
                num_to_add = num_samples - added
            else:
                num_to_add = int(num_samples / num_classes)
            new_data = i * np.ones((num_to_add, dim + 1))
            new_data[:, :dim] = np.random.multivariate_normal(
                mean=c, cov=0.75 * np.eye(dim), size=num_to_add
            )
            data[added : added + num_to_add, :] = new_data
            added += num_to_add
        np.random.shuffle(data)
        data = torch.from_numpy(data)
        return data[:, :dim].float(), data[:, dim].long()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        X, y = self.gen_dataset(self.num_samples, num_classes=3, dim=2)
        dataset = TensorDataset(X.float(), y.long())
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, self.train_val_test_split
        )
