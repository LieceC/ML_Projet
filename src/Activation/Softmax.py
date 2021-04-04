import numpy as np

from src.Module.module import Module


class Softmax(Module):
    def forward(self, X):
        e = np.exp(X)
        return e / np.sum(e, axis=1)

    def backward_delta(self, input, delta):
        ## Met a jour la valeur du gradient
        e = np.exp(input)
        sft = e / np.sum(e, axis=1)
        return delta * sft * (1 - sft)
