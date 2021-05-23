'''
Faire des tests sur les dimensions des fonctions, rapide juste un assert pour Ãªtre sur
'''
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skt

from src.Activation.sigmoid import Sigmoid
from src.Activation.softmax import Softmax
from src.Loss.CESoftMax import CESoftMax
from src.Module.linear import Linear
from src.Module.sequential import Sequential
from src.Optim.optim import Optim
from src.utils.utils import load_usps, transform_numbers


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
    validation_size = 500
    allvalx = alltestx[:validation_size]
    allvaly = alltesty[:validation_size]
    alltestx = alltestx[validation_size:]
    alltesty = alltesty[validation_size:]
    input_size = len(alltrainx[0])
    output_size = len(np.unique(alltesty))
    alltrainy_proba = transform_numbers(alltrainy, output_size)

    # Initialize modules with respective size
    iteration = 1000
    gradient_step = 1e-3
    arbitrary_neural = 128
    batch_size = 25  # len(alltrainx)

    m_linear = Linear(input_size, arbitrary_neural)
    m_act1 = Sigmoid()
    m_linear2 = Linear(arbitrary_neural, output_size)
    m_act2 = Softmax()
    m_loss = CESoftMax()

    seq = Sequential([m_linear, m_act1, m_linear2])

    opt = Optim(seq, loss=m_loss, eps=gradient_step)
    opt.SGD(alltrainx, alltrainy_proba, batch_size, X_val=allvalx, Y_val=allvaly, f_val=lambda x: np.argmax(x, axis=1),
            maxiter=iteration, verbose=2)

    predict = m_act2.forward(opt.predict(alltestx))
    predict = np.argmax(predict, axis=1)

    res = skt.confusion_matrix(predict, alltesty)
    print(np.sum(np.where(predict == alltesty, 1, 0)) / len(predict))
    plt.imshow(res)


if __name__ == '__main__':
    test_multiclass()
