import torch
from .cnn import ShallowCNN

model = ShallowCNN()
x = torch.randn(4, 3, 32, 32)
y = model(x)
print(y.shape)
