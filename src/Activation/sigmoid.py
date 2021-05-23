import numpy as np

from src.Module.module import Module


class Sigmoid(Module):
    def forward(self, X):
        return 1 / (1 + (np.exp(-X)))

    def backward_delta(self, input, delta):
        sig = 1 / (1 + np.exp(-input))
        return delta * (sig * (1 - sig))
