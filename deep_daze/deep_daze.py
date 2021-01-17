import torch
import torch.nn.functional as F
from torch import nn

class DeepDaze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
