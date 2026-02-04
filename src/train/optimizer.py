import torch.optim as optim


def get_optimizer(model):
    """
    Returns the optimizer used to train the model.
    SGD with momentum is a historically standard choice
    for CNN training.
    """
    return optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    )
