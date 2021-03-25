# -*- coding: utf-8 -*-

from module import Module
import numpy as np

class Linear(Module):
    def __init__(self,input,output):
        self._parameters = np.ones((input,output))
        self._gradient = np.zeros((input,output))
    
    def forward(self, X):
        return np.dot(X,self._parameters.T)
    
    def backward_update_gradient(self, input, delta):
        self._gradient = np.dot(input.T,delta)
        
    def backward_delta(self, input, delta):
        return np.dot(self._parameters,delta)
        
