# -*- coding: utf-8 -*-
import numpy as np

from src.Module.module import Module


class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride, pad=False):
        self._parameters = 2 * np.random.rand(k_size, chan_in, chan_out) - 1
        self._gradient = np.zeros((k_size, chan_in, chan_out))
        self._k_size = k_size
        self._stride = stride
        self._chan_in = chan_in
        self._chan_out = chan_out

    def operateur(self, X, i):
        res = np.zeros((X.shape[0], self._chan_out))
        a = X[:, i:i + self._k_size]  # cases qui nous interessent
        b = self._parameters
        for x in range(self._k_size):
            res += np.matmul(a[:, x], b[x])
        return res

    def forward(self, X):
        res = np.array([self.operateur(X, i) for i in range(0, (X.shape[1] - self._k_size) + 1, self._stride)])
        sha = res.shape
        return res.reshape((sha[1], sha[0], sha[2]))

    def operateur_update_gradient(self, X, i):
        res = np.zeros((X.shape[0], self._chan_out))
        a = X[:, i:i + self._k_size]  # cases qui nous interessent
        b = self._parameters
        for x in range(self._k_size):
            res += np.matmul(a[:, x], np.ones(b[x].shape))
        return res

    """
    def backward_update_gradient(self, input, delta):
        res = np.array(
            [self.operateur_update_gradient(input, i) for i in
             range(0, (input.shape[1] - self._k_size) + 1, self._stride)])
        sha = res.shape
        res = res.reshape((sha[1], sha[0], sha[2]))
        for x in range(input.shape[0]):
            #print(self._gradient.shape)
            self._gradient += np.dot(res[x].T, delta[x])
    """

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        if self._gradient is None:
            self._gradient = np.zeros(self._parameters.shape)

        a, _ = np.mgrid[0:input.shape[1] - self._k_size:(self._stride + 1), 0:self._k_size] \
               + np.arange(self._k_size)
        a = a.reshape(-1)
        input = input[:, a, :].reshape(input.shape[0], -1, self._k_size * self._chan_in)
        input = np.transpose(input, (1, 0, 2))
        for i in range(input.shape[0]):
            self._gradient += np.dot(input[i, :].T, delta[:, i, :]).reshape(self._gradient.shape)

    def operateur_delta(self, X, i):
        res = np.zeros((X.shape[0], self._chan_out))
        a = X[:, i:i + self._k_size]  # cases qui nous interessent
        b = self._parameters
        for x in range(self._k_size):
            res += np.matmul(np.ones(a[:, x]), b[x])
        return res

    def backward_delta(self, input, delta):
        z = zip(range(0, input.shape[1], 1 + self._stride), \
                range(self._k_size, input.shape[1], 1 + self._stride))
        res = np.zeros(input.shape)
        for i, (begin, end) in enumerate(z):
            d = np.dot(delta[:, i, :], \
                       self._parameters.reshape(-1, self._chan_out).T)
            res[:, begin:end] += d.reshape(-1, self._k_size, self._chan_in)
        return res


"""
    def backward_delta(self, input, delta):
        res = np.array(
            [self.operateur_delta(input, i) for i in range(0, (input.shape[1] - self._k_size) + 1, self._stride)])
        sha = res.shape
        res = res.reshape((sha[1], sha[0], sha[2]))
        return np.dot(delta, res)
"""

"""entre = np.random.rand(10, 128, 3)
test = Conv1D(5, 3, 4, 5)
res = test.forward(entre)
print(res)"""
