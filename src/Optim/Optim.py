import numpy as np
from src.Loss.MSELoss import MSELoss
from src.utils.utils import unison_shuffled_copies, chunks
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
        loss = self._loss.forward(batch_y, forward[-1])

        # Backward step
        backward_loss = self._loss.backward(batch_y, forward[-1])
        last_delta = self._net.backward(forward, backward_loss)

        # Update params and clean gradient.
        self._net.update_parameters(self._eps)
        self._net.zero_grad()
        return loss

    def SGD(self, X, Y, batch_size, X_val = None, Y_val = None, f_val = lambda x : x, maxiter=10, verbose=False):
        """
    
            Parameters
            ----------
            X : 
                Données d'apprentissage.
            Y : 
                Labels d'apprentissage.
            batch_size : 
                Taille des batchs pour l'apprentissage.
            X_val : TYPE, optional
                Données de validation. The default is None.
            Y_val : TYPE, optional
                Labels de validation. The default is None.
            f_val : TYPE, optional
                Fonction à appliquer aux labels predits pour correspondre aux labels des données de validation. 
                The default is lambda x : x.
            maxiter : TYPE, optional
                Nombre d'itération d'apprentissage. The default is 10.
            verbose : TYPE, optional
                Parametre de verbosité.
                1 => affiche loss chaque itération
                2 => affiche la courbe de l'évolution de la loss en fonction du nombre d'itérations
                The default is False.
            Returns
            -------
            None.

        """
        assert len(X) == len(Y)
        assert X_val is None or (Y_val is not None and len(X_val) == len(Y_val))
        losss = []
        precision_val = []
        for i in range(maxiter):
            datax_rand, datay_rand = unison_shuffled_copies(X, Y)
            datax_rand_batch, datay_rand_batch = list(chunks(datax_rand, batch_size)), list(
                chunks(datay_rand, batch_size))
            nb_batchs = len(datax_rand_batch)
            loss_batch = np.zeros(nb_batchs)

            for j in range(nb_batchs):
                loss_batch[j] = self.step(datax_rand_batch[j], datay_rand_batch[j]).mean()
                losss += [loss_batch[j]]
            if X_val is not None: # calcul validation
                predict = self.predict(X_val)
                y_hat = f_val(predict)
                precision_val += [sum(y_hat == Y_val)/len(Y_val)]
                
            if verbose >= 1: 
                print("iteration "+str(i)+":")
                print("Loss")
                print("mean - "+str(loss_batch.mean()) + "\nstd - "+str(loss_batch.std())) 
        if verbose == 2:
            patches = [mpatches.Patch(color='red', label = 'variance sur iterations'),
                       mpatches.Patch(color='green', label = 'moyenne sur iterations'),
                       mpatches.Patch(color='blue', label = 'evolution sur les batchs')
                       ]
            losss = np.array(losss)
            x = np.arange(1/nb_batchs,maxiter+1/nb_batchs,1/nb_batchs)
            plt.plot(x,losss,color="blue")
            plt.title("Evolution de la loss en fonction du nombre d'itérations")
            plt.legend(handles=patches[-1:])
            plt.show()
            
            x = np.arange(1,maxiter+1/nb_batchs,1)
            losss_2 = losss.reshape(-1,nb_batchs)
            plt.plot(x,losss_2.mean(axis=1),color='green')
            plt.plot(x,losss_2.std(axis=1),color='red')
            plt.title("Evolution de la loss en fonction du nombre d'itérations")
            plt.legend(handles=patches[:-1])
            plt.show()
        if X_val is not None:
            x = np.arange(1,maxiter+1)
            plt.plot(x,precision_val,color="blue")
            plt.title("Evolution de la précision en fonction du nombre d'itérations")
            plt.show()

    def predict(self, X):
        return self._net.forward(X)[-1]
