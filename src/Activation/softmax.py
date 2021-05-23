import numpy as np

from src.Module.module import Module


class Softmax(Module):
    def forward(self, X):
        e = np.exp(X)
        return e / np.sum(e, axis=1)[..., None]

    def backward_delta(self, input, delta):
        e = np.exp(input)
        sft = e / np.sum(e, axis=1)[..., None]
        return delta * sft * (1 - sft)
