'''
Faire des tests sur les dimensions des fonctions, rapide juste un assert pour être sur
'''
import numpy as np

import src.utils.mltools as tools
from src.Loss.MSELoss import MSELoss
from src.Module.linear import Linear


def test_dim_linear():
    # generation of tests data
    datax, datay = tools.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type=0, epsilon=0.1)
    datay_r = np.zeros((len(datay), 2))
    # Re-arranging data to compute a probability
    for y in range(len(datay)):
        if datay[y] == -1:
            datay_r[y][0] = 1
        elif datay[y] == 1:
            datay_r[y][1] = 1

    # Input and Output size of our NN
    input_size = len(datax[0])
    output_size = len(np.unique(datay))
    # Initialize modules with respective size
    m_mse = MSELoss()
    m_linear = Linear(input_size, output_size)
    # Etape forward
    hidden_l = m_linear.forward(datax)
    assert (hidden_l.shape == (len(datax), output_size))
    loss = m_mse.forward(hidden_l, datay_r)
    assert (loss.shape == (len(datax),))
    # Etape Backward
    loss_back = m_mse.backward(hidden_l, datay_r)
    assert (loss_back.shape == (len(datax), output_size))
    delta_linear = m_linear.backward_delta(datax, loss_back)
    assert (delta_linear.shape == (len(datax), output_size))


if __name__ == '__main__':
    test_dim_linear()
