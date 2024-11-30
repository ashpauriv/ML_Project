import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch import optim
"""
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from touchvision import datasets
from torchvision.transforms import ToTensor
"""

# Define dataset
class dataset(data):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32) # not sure what dtype does
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# Define the model
class UbervsLift_relu(nn.Module):
    def __init__(self, input_size, num_classes):
        super()
    def forward(self, x):
        # implement forward
        return x


