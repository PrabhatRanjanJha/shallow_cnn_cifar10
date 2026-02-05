import torch
from src.train.loss import get_loss
from src.train.optimizer import get_optimizer

from src.model.cnn import ShallowCNN
from src.data.dataloader import get_train_dataloader, get_test_dataloader

import csv
import os


def train():
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # -------- Device configuration --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Data loaders --------
    train_loader = get_train_dataloader(batch_size=64)
    test_loader = get_test_dataloader(batch_size=64)

    # -------- Model --------
    model = ShallowCNN().to(device)

    # -------- Loss function --------
    criterion = get_loss()

    # -------- Optimizer (historically correct choice) --------
    optimizer = get_optimizer(model)

    # -------- Training loop --------
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()  # set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 1. Zero previous gradients
            optimizer.zero_grad()

            # 2. Forward pass
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 3. Compute loss
            loss = criterion(outputs, labels)

            # 4. Backward pass
            loss.backward()

            # 5. Update weights
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        
        epoch_acc = 100 * correct / total
        train_accuracies.append(epoch_acc)
        train_losses.append(avg_loss)
        
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_loss:.4f}, Train Accuracy: {epoch_acc:.2f} | Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}")

    print("Training finished.")
    torch.save(model.state_dict(), "model.pth")

    os.makedirs("experiments", exist_ok=True)

    with open("experiments/training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        for i in range(num_epochs):
            writer.writerow([i+1, train_losses[i], train_accuracies[i], val_losses[i], val_accuracies[i]])



def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


if __name__ == "__main__":
    train()
    
