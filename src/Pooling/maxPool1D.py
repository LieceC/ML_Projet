import numpy as np

from src.Module.module import Module


class MaxPool1D(Module):
    def forward(self, X):
        return np.max(0.01*X, X)

    def backward_delta(self, input, delta):
        return delta * np.where(input < 0, 0.01, 1)
