# -*- coding: utf-8 -*-

from loss import Loss
import numpy as np

class MSELoss(Loss):
    def forward(self, y, yhat):
        '''
        y et yhat de taille (batch,d)
        batch : nombre d'exemples
        d : nombre de classes
        '''
        return np.linalg.norm(y-yhat,axis=1)**2
        
    def backward(self, y, yhat):
        return -2*(y-yhat)