def accuracy(predictions, labels):
    """
    Computes the number of correct predictions in a batch.
    """
    correct = (predictions == labels).sum().item()
    return correct
