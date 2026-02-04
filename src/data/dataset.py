from torchvision.datasets import CIFAR10
from .transforms import train_transforms, test_transforms


def get_train_dataset(data_dir="data/raw"):
    """
    Returns the CIFAR-10 training dataset with training transforms applied.
    """
    train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transforms
    )
    return train_dataset


def get_test_dataset(data_dir="data/raw"):
    """
    Returns the CIFAR-10 test dataset with test transforms applied.
    """
    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transforms
    )
    return test_dataset
