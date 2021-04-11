import numpy as np

from src.Module.module import Module


class ReLU(Module):
    def forward(self, X):
        return np.max(0, X)

    def backward_delta(self, input, delta):
        return delta * np.where(input < 0, 0, 1)
