import numpy as np

from src.Module.module import Module


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        self._k_size = k_size
        self._stride = stride

    def forward(self, X):
        array_pool = np.zeros(((len(X) - self.k_size) / self.stride + 1), len(X[0]))
        for x in range(0, len(array_pool), self._stride):
            array_pool[x] = np.max(X[x * self._stride, x * self._stride + self._k_size + 1])
        return array_pool

    def backward_delta(self, input, delta):
        new_size = (len(input[0]) - 1) * self._stride + self._k_size
        array_pool = np.zeros(new_size)
        for x in range(0, new_size, self._k_size):
            array_pool[x:x * self._k_size] = delta * input[x]
        return array_pool
