import csv
import matplotlib.pyplot as plt


def load_training_log(csv_path):
    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_accuracy"]))
            val_acc.append(float(row["val_accuracy"]))

    return epochs, train_loss, val_loss, train_acc, val_acc


def plot_loss(epochs, train_loss, val_loss):
    plt.figure()
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy(epochs, train_acc, val_acc):
    plt.figure()
    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    log_path = "experiments/training_log.csv"

    epochs, train_loss, val_loss, train_acc, val_acc = load_training_log(log_path)

    plot_loss(epochs, train_loss, val_loss)
    plot_accuracy(epochs, train_acc, val_acc)