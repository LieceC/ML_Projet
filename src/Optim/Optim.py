from src.Loss.MSELoss import MSELoss
from src.utils.utils import unison_shuffled_copies, chunks
import numpy as np

class Optim(object):
    """
    Assum that net is a sequential class
    """

    def __init__(self, net, loss=MSELoss(), eps=1e-3):
        self._loss = loss
        self._eps = eps
        self._net = net

    # SGD step
    def step(self, batch_x, batch_y):
        # Forward step
        forward = self._net.forward(batch_x)
        loss = self._loss.forward(batch_y,forward[-1])

        # Backward step
        backward_loss = self._loss.backward(batch_y, forward[-1])
        last_delta = self._net.backward(forward, backward_loss)

        # Update params and clean gradient.
        self._net.update_parameters(self._eps)
        self._net.zero_grad()
        return loss

    def SGD(self, X, Y, batch_size, maxiter=10, verbose = False):
        assert len(X) == len(Y)
        for i in range(maxiter):
            datax_rand, datay_rand = unison_shuffled_copies(X, Y)
            datax_rand_batch, datay_rand_batch = list(chunks(datax_rand, batch_size)), list(
                chunks(datay_rand, batch_size))
            nb_batchs = len(datax_rand_batch)
            losss = np.zeros(nb_batchs)
            for j in range(nb_batchs):
                losss[j] = self.step(datax_rand_batch[j], datay_rand_batch[j]).mean()
            if verbose == 1: 
                print("iteration "+str(i)+":")
                print("Loss")
                print("mean - "+str(losss.mean()) + "\nstd - "+str(losss.std())) 
                
    def predict(self, X):
        return self._net.forward(X)[-1]
