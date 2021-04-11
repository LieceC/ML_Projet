# -*- coding: utf-8 -*-
import numpy as np
from src.Module.module import Module

class Conv1D(Module):
    def __init__(self,k_size, chan_in, chan_out, stride, pad = False):
        self._parameters = 2 * np.random.rand(k_size, chan_in, chan_out) - 1
        self._k_size = k_size
        self._stride = stride
        self._chan_in = chan_in
        self._chan_out = chan_out
        
    def operateur(self,X,i):
        res = np.zeros((X.shape[0],self._chan_out))
        a = X[:,i:i+self._k_size] # cases qui nous interessent
        b = self._parameters
        for x in range(self._k_size):
            res += np.matmul(a[:,x],b[x])
        return res
    
    def forward(self,X):
        res = np.array([self.operateur(X, i) for i in range(0,X.shape[1] - self._k_size,self._stride)])
        sha = res.shape
        return res.reshape((sha[1],sha[0],sha[2]))
    
    def operateur_update_gradient(self,X,i):
        res = np.zeros((X.shape[0],self._chan_out))
        a = X[:,i:i+self._k_size] # cases qui nous interessent
        b = self._parameters
        for x in range(self._k_size):
            res += np.matmul(a[:,x],np.ones(b[x].shape))
        return res
    
    def backward_update_gradient(self, input, delta):
        res = np.array([self.operateur_update_gradient(input, i) for i in range(0,input.shape[1] - self._k_size,self._stride)])
        sha = res.shape
        res = res.reshape((sha[1],sha[0],sha[2]))
        self._gradient += np.dot(res.T, delta)
        if self._bias:
            # print(np.sum(delta, axis = 0))
            self._bias_gradient += np.sum(delta, axis=0)
    
    def operateur_delta(self,X,i):
        res = np.zeros((X.shape[0],self._chan_out))
        a = X[:,i:i+self._k_size] # cases qui nous interessent
        b = self._parameters
        for x in range(self._k_size):
            res += np.matmul(np.ones(a[:,x]),b[x])
        return res
    
    def backward_delta(self, input, delta):
        res = np.array([self.operateur_delta(input, i) for i in range(0,input.shape[1] - self._k_size,self._stride)])
        sha = res.shape
        res = res.reshape((sha[1],sha[0],sha[2]))
        return np.dot(delta, res)
    
entre = np.random.rand(10,128,3)
test = Conv1D(5,3,4,5)
res = test.forward(entre)
print(res)