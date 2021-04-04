'''
Faire des tests sur les dimensions des fonctions, rapide juste un assert pour Ãªtre sur
'''
import numpy as np
import sklearn.metrics as skt

from src.Activation.Softmax import Softmax
from src.Activation.sigmoid import Sigmoid
from src.Loss.BCE import BCE
from src.Loss.MSELoss import MSELoss
from src.Module.Linear import Linear
from src.utils.utils import load_usps
import matplotlib.pyplot as plt 

def transform_numbers(input, size):
    """Assume 1D array as input, len is the number of example
    Transform into proba
    """
    datay_r = np.zeros((len(input), size))
    # Re-arranging data to compute a probability
    for x in range(len(input)):
        datay_r[x][input[x]] = 1
    return datay_r


def test_multiclass():
    """
    testx, testy = get_usps([neg, pos], alltestx, alltesty)
    testy = np.where(testy == neg, -1, 1)
    :return:
    """
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)
    input_size = len(alltrainx[0])
    output_size = len(np.unique(alltesty))
    alltrainy_proba = transform_numbers(alltrainy, output_size)

    # Initialize modules with respective size
    iteration = 30
    gradient_step = 10e-5
    arbitrary_neural = 128
    m_linear = Linear(input_size, arbitrary_neural)
    m_act1 = Sigmoid()
    m_linear2 = Linear(arbitrary_neural, output_size)
    m_act2 = Softmax()
    m_loss = BCE()

    for _ in range(iteration):
        # Etape forward
        hidden_l1 = m_linear.forward(alltrainx)
        act1 = m_act1.forward(hidden_l1)
        hidden_l2 = m_linear2.forward(act1)
        act2 = m_act2.forward(hidden_l2)
        loss = m_loss.forward(alltrainy_proba,act2)
        print("max loss:", np.max(loss))
        # print("parameters",m_linear._parameters)
        # Etape Backward
        loss_back = m_loss.backward(alltrainy_proba, act2)
        act2_back = m_act2.backward_delta(hidden_l2, loss_back)
        hidden_l2_back = m_linear2.backward_delta(act1, act2_back)
        act1_back = m_act1.backward_delta(hidden_l1, hidden_l2_back)
        hidden_l1_back = m_linear.backward_delta(alltrainx, act1_back)

        #update gradient
        m_linear2.backward_update_gradient(act1, act2_back)
        m_linear.backward_update_gradient(alltrainx, act1_back)

        # update parameters
        m_linear2.update_parameters(gradient_step=gradient_step)
        m_linear.update_parameters(gradient_step=gradient_step)

        m_linear.zero_grad()
        m_linear2.zero_grad()

    hidden_l1 = m_linear.forward(alltestx)
    act1 = m_act1.forward(hidden_l1)
    hidden_l2 = m_linear2.forward(act1)
    act2 = m_act2.forward(hidden_l2)
    predict = np.argmax(act2, axis=1)

    res = skt.confusion_matrix(predict, alltesty)
    plt.imshow(res)

if __name__ == '__main__':
    test_multiclass()
