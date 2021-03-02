from torchvision import datasets, transforms
import sys
print(sys.path)
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, target_transform=None):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm, target_transform=None)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MnistOneVsAllDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    sets = []

    def __init__(self, data_dir, batch_size, shuffle=True,
                 validation_split=0.0, num_workers=1, training=True):
        for i in range(10):
            data_loader_i = MnistDataLoader(data_dir, batch_size, shuffle,
                                            validation_split, num_workers,
                                            training,
                                            lambda x: 1 if x == i else 0)
            self.sets.append(data_loader_i)
        return self.sets
