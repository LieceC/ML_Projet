# -*- coding: utf-8 -*-

import numpy as np

from src.Module.module import Module


class Linear(Module):
    def __init__(self, input, output, bias=True):
        self._parameters = 2 * np.random.rand(input, output) - 1
        self._gradient = np.zeros((input, output))
        self._bias = bias
        if self._bias:
            self._bias_parameters = 2 * np.random.rand(1, output) - 1
            self._bias_gradient = np.zeros((1, output))

    def forward(self, X):
        if self._bias:
            return np.dot(X, self._parameters) + self._bias_parameters
        return np.dot(X, self._parameters)

    def backward_update_gradient(self, input, delta):
        self._gradient += np.dot(input.T, delta)
        if self._bias:
            self._bias_gradient += np.sum(delta, axis=0)

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient
        if self._bias:
            self._bias_parameters -= gradient_step * self._bias_gradient

    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)
        if self._bias:
            self._bias_gradient = np.zeros(self._bias_gradient.shape)

    def backward_delta(self, input, delta):
        return np.dot(delta, self._parameters.T)
