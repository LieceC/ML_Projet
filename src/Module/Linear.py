# -*- coding: utf-8 -*-

import numpy as np

from src.Module.module import Module

optim_list = ["SGD", "SGD_Momentum", "ADAGrad", "RMSProp", "ADAM"]


class Linear(Module):
    def __init__(self, input, output, gradient_method="SGD", rho=0.99, decay_rate = 0.5, bias=True):
        self._parameters = 2 * np.random.rand(input, output) - 1
        print(self._parameters)
        self._gradient = np.zeros((input, output))
        self._gradient_method = gradient_method
        self._bias = bias
        self._rho = rho
        self._decay_rate = 0.5
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
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        if self._gradient_method == "SGD":
            self._parameters -= gradient_step * self._gradient
            if self._bias:
                self._bias_parameters -= gradient_step * self._bias_gradient
        elif self._gradient_method == "SGD_Momentum":
            if not hasattr(self, 'vx'):
                self._vx = 0
            self._vx = self._rho * self._vx + self._gradient
            self._parameters -= self._vx * self._gradient
            if self._bias:
                self._bias_parameters -= self._vx * self._bias_gradient
        elif self._gradient_method == "ADAGrad":
            if not hasattr(self, 'sGrad'):
                self._sGrad = 0
                self._sGrad_bias = 0  # ???
            self._sGrad += self._gradient * self._gradient
            self._parameters -= gradient_step * self._gradient / (np.sqrt(self._sGrad) + 1e-7)
            if self._bias:
                self._sGrad_bias += self._gradient_bias * self._gradient_bias
                self._bias_parameters -= gradient_step * self._gradient_bias / (np.sqrt(self._sGrad_bias) + 1e-7)
        #To continue
        #elif self._gradient_method == "RMSProp":


    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)
        if self._bias:
            self._bias_gradient = np.zeros(self._bias_gradient.shape)

    def backward_delta(self, input, delta):
        return np.dot(delta, self._parameters.T)
