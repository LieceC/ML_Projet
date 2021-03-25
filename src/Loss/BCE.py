import numpy as np

from loss import Loss


class BCE(Loss):
    def forward(self, y, yhat):
        '''
        y et yhat de taille (batch,d)
        batch : nombre d'exemples
        d : nombre de classes
        '''
        # TODO Verifier la dimension de sortie
        return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    def backward(self, y, yhat):
        return (y - yhat) / ((yhat - 1) * yhat)