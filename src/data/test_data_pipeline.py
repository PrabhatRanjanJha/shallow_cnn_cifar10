from .dataloader import get_train_dataloader, get_test_dataloader

train_loader = get_train_dataloader(batch_size=64)
test_loader = get_test_dataloader(batch_size=64)

images, labels = next(iter(train_loader))

print("Train batch image shape:", images.shape)
print("Train batch label shape:", labels.shape)
print("Image dtype:", images.dtype)
print("Label dtype:", labels.dtype)
print("Min pixel value:", images.min().item())
print("Max pixel value:", images.max().item())
print("Mean pixel value:", images.mean().item())
