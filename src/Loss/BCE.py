import numpy as np

from src.Loss.loss import Loss


class BCE(Loss):
    def forward(self, y, yhat):
        """
        input  : batch*d
        output : batch
        """
        a = np.log(yhat)
        a = np.where(a < -100, -100, a)
        b = np.log(1 - yhat)
        b = np.where(np.isnan(b), -100, b)

        return - (y * a + (1 - y) * b)

    def backward(self, y, yhat, eps=1e-10):
        return ((1 - y) / (1 - yhat + eps)) - (y / (yhat + eps))
