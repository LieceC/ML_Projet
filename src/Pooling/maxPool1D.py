import numpy as np

from src.Module.module import Module


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        self._k_size = k_size
        self._stride = stride
        

    def forward(self, X):
        array_pool = np.zeros(((X.shape[0],X.shape[1] - self.k_size) // self.stride + 1), X.shape[2])
        
        self._argmax_array = np.zeros(((X.shape[0],X.shape[1] - self.k_size) // self.stride + 1), X.shape[2])
        
        for x in range(0, array_pool.shape[1]):
            array_pool[:,x] = np.max(X[:,x * self._stride: x * self._stride + self._k_size], axis = 1)
            self._argmax_array[:,x] = np.argmax(X[:,x * self._stride: x * self._stride + self._k_size], axis = 1)
        return array_pool

    def backward_delta(self, input, delta):
        new_size = (input.shape[1] - 1) * self._stride + self._k_size
        array_pool = np.zeros((input.shape[0],new_size))
        for x in range(0, new_size):
            array_pool[:,x] = delta[self._argmax_array[:,x]]
        return array_pool
    '''
    def backward_delta(self, input, delta):
        new_size = (input.shape[1] - 1) * self._stride + self._k_size
        array_pool = np.zeros((input.shape[0],new_size))
        for x in range(0, new_size):
            array_pool[:,x:x * self._k_size] = delta * input
        return array_pool
    '''