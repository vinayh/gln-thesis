from typing import Optional, Tuple

from torch.utils.data import ConcatDataset, random_split
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import transforms
from torch import Generator
from os.path import join

from src.datamodules.mnist_utils import divide_fn, deskew_fn
from src.datamodules.pretrain_datamodule import PretrainDataModule

# SEED = 42
SEED = 17

transforms_empty = transforms.Compose([transforms.ToTensor()])

transforms_default = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(divide_fn), transforms.ToTensor()]
)

transforms_deskew = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(deskew_fn), transforms.ToTensor()]
)

transforms_normalize = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


class MNISTDataModule(PretrainDataModule):
    def __init__(
        self,
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        deskew: bool = False,
        fashionmnist: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dataset_name = "mnist"
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if fashionmnist:
            self.dataset_fn = FashionMNIST
            print("MNIST DataModule: using Fashion MNIST")
        else:
            self.dataset_fn = MNIST

        if deskew:
            print("MNIST DataModule: deskewing enabled")
            self.transforms = transforms_deskew
        else:
            self.transforms = transforms_empty
            # self.transforms = transforms.Compose([transforms.ToTensor()])

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, 28, 28)

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        self.dataset_fn(self.data_dir, train=True, download=True)
        self.dataset_fn(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        trainset = self.dataset_fn(
            self.data_dir, train=True, download=True, transform=self.transforms
        )
        testset = self.dataset_fn(
            self.data_dir, train=False, download=True, transform=self.transforms
        )
        dataset = ConcatDataset(datasets=[trainset, testset])
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, self.train_val_test_split, generator=Generator().manual_seed(SEED)
        )
