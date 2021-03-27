'''
Faire des tests sur les dimensions des fonctions, rapide juste un assert pour Ãªtre sur
'''
import numpy as np

import src.utils.mltools as tools
from src.Loss.BCE import BCE
from src.Module.Linear import Linear


def test_linear():
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
    
    
    
    # Input and Output size of our NN
    input_size = len(datax[0])
    output_size = len(np.unique(datay))
    
    # Initialize modules with respective size
    iteration = 10
    gradient_step = 10e-8
    m_bce = BCE()
    m_linear = Linear(input_size, output_size)
        
    for _ in range(iteration):
       
        # Etape forward
        hidden_l = m_linear.forward(datax)
        loss = m_bce.forward(hidden_l, datay_r)
        print("max loss:",np.max(loss))
        # print("parameters",m_linear._parameters)
        # Etape Backward
        loss_back = m_bce.backward(hidden_l, datay_r)
        delta_linear = m_linear.backward_delta(datax, loss_back)
        
        m_linear.backward_update_gradient(datax, delta_linear)
        # print("gradient",m_linear._gradient)
        m_linear.update_parameters(gradient_step = gradient_step)
        m_linear.zero_grad()
    '''
    res = m_linear.forward(testx)
    res = m_mse.forward(res, testy_r)
    print(res)
    res = np.argmax(res,axis=1)
    print(res)
    '''
if __name__ == '__main__':
    test_linear()
