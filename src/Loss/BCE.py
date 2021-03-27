import numpy as np

from src.Loss.loss import Loss


class BCE(Loss):
    def forward(self, y, yhat, eps=10e-10):
        '''
        y et yhat de taille (batch,d)
        batch : nombre d'exemples
        d : nombre de classes
        '''

        # TODO Verifier la dimension de sortie
        return -(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))

    def backward(self, y, yhat, eps=10e-10):
        return -((y - yhat) / ((yhat - 1) * yhat + eps))
