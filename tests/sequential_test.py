'''
Faire des tests sur les dimensions des fonctions, rapide juste un assert pour Ãªtre sur
'''
import numpy as np
import sklearn.metrics as skt

from src.Activation.Softmax import Softmax
from src.Activation.sigmoid import Sigmoid
from src.Activation.tanH import TanH
from src.Loss.BCE import BCE
from src.Loss.MSELoss import MSELoss
from src.Loss.CESoftMax import CESoftMax
from src.Module.Linear import Linear
from src.utils.utils import load_usps
from src.Module.sequential import Sequential
from src.Optim.Optim import Optim
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
    iteration = 100
    gradient_step = 1e-3
    arbitrary_neural = 128
    batch_size = 100 # len(alltrainx)
    
    m_linear = Linear(input_size, arbitrary_neural)
    m_act1 = Sigmoid()
    m_linear2 = Linear(arbitrary_neural, output_size)
    m_act2 = Softmax()
    m_loss = CESoftMax()
    
    seq = Sequential([m_linear,m_act1,m_linear2])
    
    opt = Optim(seq,loss=m_loss,eps = gradient_step)
    opt.SGD(alltrainx,alltrainy_proba,batch_size, maxiter=iteration,verbose = True)

    predict = m_act2.forward(opt.predict(alltestx))
    predict = np.argmax(predict, axis=1)

    res = skt.confusion_matrix(predict, alltesty)
    print(np.sum(np.where(predict==alltesty,1,0))/len(predict))
    plt.imshow(res)

if __name__ == '__main__':
    test_multiclass()
