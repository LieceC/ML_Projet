'''
Faire des tests sur les dimensions des fonctions, rapide juste un assert pour Ãªtre sur
'''
import numpy as np

from src.Loss.MSELoss import MSELoss
from src.Module.Linear import Linear


def test_linear():
    coef1 = 58
    coef2 = 24

    # fonction linéair que l'on apprend
    def f(x1, x2):
        return x1 * coef1 - coef2 * x2

    # données d'entrainement avec bruit
    def f_bruit(x1, x2):
        bruit = np.random.normal(0, 1, len(x1)).reshape((-1, 1))
        return f(x1, x2) + bruit

    '''
    # generation of tests data
    datax, datay = tools.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type=0, epsilon=0.1)
    testx, testy = tools.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type=0, epsilon=0.1)
    
    datay_r = np.zeros((len(datay), 2))
    # Re-arranging data to compute a probability
    for y in range(len(datay)):
        if datay[y] == -1:
            datay_r[y][0] = 1
        elif datay[y] == 1:
            datay_r[y][1] = 1
    testy_r = np.zeros((len(datay), 2))
    for y in range(len(testy)):
        if testy[y] == -1:
            testy_r[y][0] = 1
        elif testy[y] == 1:
            testy_r[y][1] = 1
    '''
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
    gradient_step = 10e-5
    m_mse = MSELoss()
    m_linear = Linear(input_size, output_size)

    for _ in range(iteration):
        # Etape forward
        hidden_l = m_linear.forward(datax)
        loss = m_mse.forward(hidden_l, datay)
        print("max loss:", np.max(loss))
        # print("parameters",m_linear._parameters)
        # Etape Backward

        loss_back = m_mse.backward(datay, hidden_l)
        # print(loss_back)
        # print(m_linear._parameters)
        delta_linear = m_linear.backward_delta(datax, loss_back)

        m_linear.backward_update_gradient(datax, loss_back)
        # print("gradient",m_linear._gradient)
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


if __name__ == '__main__':
    test_linear()
