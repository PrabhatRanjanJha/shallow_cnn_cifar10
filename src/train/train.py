import torch
import torch.nn as nn
import torch.optim as optim

from src.model.cnn import ShallowCNN
from src.data.dataloader import get_train_dataloader, get_test_dataloader


def train():
    # -------- Device configuration --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Data loaders --------
    train_loader = get_train_dataloader(batch_size=64)
    test_loader = get_test_dataloader(batch_size=64)

    # -------- Model --------
    model = ShallowCNN().to(device)

    # -------- Loss function --------
    criterion = nn.CrossEntropyLoss()

    # -------- Optimizer (historically correct choice) --------
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    )

    # -------- Training loop --------
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()  # set model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 1. Zero previous gradients
            optimizer.zero_grad()

            # 2. Forward pass
            outputs = model(images)

            # 3. Compute loss
            loss = criterion(outputs, labels)

            # 4. Backward pass
            loss.backward()

            # 5. Update weights
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    train()
