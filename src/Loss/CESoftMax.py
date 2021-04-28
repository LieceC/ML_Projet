import numpy as np

from src.Loss.loss import Loss


class CESoftMax(Loss):
    def forward(self, y, yhat, eps=10e-10):
        '''
        y et yhat de taille (batch,d)
        batch : nombre d'exemples
        d : nombre de classes
        '''
        return - np.sum(y * yhat, axis=1) + np.log(np.sum(np.exp(yhat), axis=1))

    def backward(self, y, yhat, eps=10e-10):
        e = np.exp(yhat)
        return - y + e / np.sum(e, axis=1)[..., None]


class CE(Loss):
    def forward(self, y, yhat, eps=10e-10):
        '''
        y et yhat de taille (batch,d)
        batch : nombre d'exemples
        d : nombre de classes
        '''
        return - np.sum(y * yhat, axis=1)

    def backward(self, y, yhat, eps=10e-10):
        return - y
