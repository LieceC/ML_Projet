import numpy as np

from src.Loss.loss import Loss


class CESoftMax(Loss):
    def forward(self, y, yhat, eps=10e-10):
        '''
        y et yhat de taille (batch,d)
        batch : nombre d'exemples
        d : nombre de classes
        '''
        return y * yhat + np.log(np.sum(np.exp(yhat * y), axis=1))

    def backward(self, y, yhat, eps=10e-10):
        return y + np.sum(np.exp(yhat * y), axis=1) / yhat * y * np.sum(np.exp(yhat * y))
