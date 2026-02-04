import torch.nn as nn


def get_loss():
    """
    Returns the loss function used for training.
    CrossEntropyLoss is suitable for multi-class classification
    and expects raw logits from the model.
    """
    return nn.CrossEntropyLoss()
