from torch.utils.data import DataLoader
from .dataset import get_train_dataset, get_test_dataset


def get_train_dataloader(
    data_dir="data/raw",
    batch_size=64,
    num_workers=2
):
    """
    Returns DataLoader for CIFAR-10 training set.
    """
    train_dataset = get_train_dataset(data_dir)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_loader


def get_test_dataloader(
    data_dir="data/raw",
    batch_size=64,
    num_workers=2
):
    """
    Returns DataLoader for CIFAR-10 test set.
    """
    test_dataset = get_test_dataset(data_dir)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_loader
