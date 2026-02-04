import torch

from src.model.cnn import ShallowCNN
from src.data.dataloader import get_test_dataloader
from src.evaluate.metrics import accuracy


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_test_dataloader(batch_size=64)

    model = ShallowCNN().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += accuracy(predicted, labels)

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    evaluate()
