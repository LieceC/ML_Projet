import numpy as np

from src.Module.module import Module


class TanH(Module):
    def forward(self, X):
        return np.tanh(X)

    def backward_delta(self, input, delta):
        tan2 = np.tanh(input)
        return delta * (1 - tan2 * tan2)
