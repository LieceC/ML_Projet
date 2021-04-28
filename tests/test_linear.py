'''
Faire des tests sur les dimensions des fonctions, rapide juste un assert pour Ãªtre sur
'''
import numpy as np

from src.Loss.MSELoss import MSELoss
from src.Module.Linear import Linear


def test_linear():
    coef1 = 1002
    coef2 = 13
    bias = 4

    # fonction linéaire que l'on apprend
    def f(x1, x2):
        return x1 * coef1 + coef2 * x2 + bias

    # données d'entrainement avec bruit
    def f_bruit(x1, x2):
        bruit = np.random.normal(0, 1, len(x1)).reshape((-1, 1))
        return f(x1, x2) + bruit

    nb_data = 100
    x1 = np.random.uniform(-10, 10, nb_data)
    x2 = np.random.uniform(-10, 10, nb_data)
    x1 = x1.reshape((-1, 1))
    x2 = x2.reshape((-1, 1))
    datay = f_bruit(x1, x2)

    datax = np.concatenate((x1, x2), axis=1)
    # Input and Output size of our NN
    input_size = len(datax[0])
    output_size = 1

    # Initialize modules with respective size
    iteration = 30
    gradient_step = 1e-3
    m_mse = MSELoss()
    m_linear = Linear(input_size, output_size)

    for _ in range(iteration):
        hidden_l = m_linear.forward(datax)
        loss = m_mse.forward(datay, hidden_l)
        print("max loss:", np.max(loss))
        loss_back = m_mse.backward(datay, hidden_l)

        m_linear.backward_update_gradient(datax, loss_back)
        m_linear.update_parameters(gradient_step=gradient_step)
        m_linear.zero_grad()

    x1 = np.random.uniform(-10, 10, nb_data)
    x2 = np.random.uniform(-10, 10, nb_data)
    x1 = x1.reshape((-1, 1))
    x2 = x2.reshape((-1, 1))
    testy = f(x1, x2)
    testx = np.concatenate((x1, x2), axis=1)
    hidden_l = m_linear.forward(testx)
    print("max différence res:", np.max(hidden_l - testy))
    print("parameters:", str(m_linear._parameters))
    print("valeurs voulues:", str([[coef1], [coef2]]))
    print("biais:", str(m_linear._bias_parameters))
    print("valeurs voulues:", str([bias]))


def test_linear_SGD():
    pass


if __name__ == '__main__':
    test_linear()
    test_linear_SGD()
