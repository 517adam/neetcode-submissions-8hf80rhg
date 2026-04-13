import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.linear = nn.Linear(784,512)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.act2 = nn.Sigmoid()
        self.proj = nn.Linear(512,10)

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x = self.linear(images)
        x = self.act(x)
        x = self.dropout(x)
        out = self.act2(self.proj(x))
        return out