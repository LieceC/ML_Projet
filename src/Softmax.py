import numpy as np

from module import Module


class Softmax(Module):
    def __init__(self):
        super.__init__(self)

    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)

    def forward(self, X):
        return np.exp(X) / np.sum(np.exp(X), axis=0)  # Ou axis = 1 ?

    def update_parameters(self, gradient_step=1e-3):
        # Pas de parametres dans une fonction d'activation
        pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass


    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass
