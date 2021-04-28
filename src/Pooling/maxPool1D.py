import numpy as np

from src.Module.module import Module


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self._k_size = k_size
        self._stride = stride

    def forward(self, X):
        size = ((X.shape[1] - self._k_size) // self._stride) + 1
        array_pool = np.zeros((X.shape[0], size, X.shape[2]))
        self._argmax_array = np.zeros((X.shape[0], size, X.shape[2]))

        for x in range(0, array_pool.shape[1]):
            array_pool[:, x] = np.max(X[:, (x * self._stride): (x * self._stride + self._k_size)], axis=1)
            self._argmax_array[:, x] = np.argmax(X[:, (x * self._stride): (x * self._stride + self._k_size)],
                                                 axis=1) + x * self._stride
        self._argmax_array = np.intc(self._argmax_array)
        return array_pool

    def backward_delta(self, input, delta):
        res = np.zeros(input.shape)
        for i in range(self._argmax_array.shape[1]):
            idx = self._argmax_array[:, i, :].flatten()
            res[np.repeat(range(input.shape[0]), input.shape[2]), idx, list(range(input.shape[2])) * input.shape[
                0]] = delta[:, i, :].flatten()
        return res
