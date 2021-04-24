# -*- coding: utf-8 -*-

'''
Faire des tests sur les dimensions des fonctions, rapide juste un assert pour Ãªtre sur
'''
import numpy as np

from src.Activation.ReLU import ReLU
from src.Loss.CESoftMax import CESoftMax
from src.Module.Conv1D import Conv1D
from src.Module.Linear import Linear
from src.Module.flatten import Flatten
from src.Module.sequential import Sequential
from src.Optim.Optim import Optim
from src.Pooling.maxPool1D import MaxPool1D
from src.utils.utils import load_usps
from src.Activation.Softmax import Softmax


def transform_numbers(input, size):
    """Assume 1D array as input, len is the number of example
    Transform into proba
    """
    datay_r = np.zeros((len(input), size))
    # Re-arranging data to compute a probability
    for x in range(len(input)):
        datay_r[x][input[x]] = 1
    return datay_r

if __name__ == '__main__':
    # Get the data
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)
    alltrainy_proba = transform_numbers(alltrainy, np.unique(alltrainy).shape[0])
    alltrainx = alltrainx.reshape((alltrainx.shape[0], alltrainx.shape[1], 1))
    alltestx = alltestx.reshape((alltestx.shape[0], alltestx.shape[1], 1))
    # Get data values
    length = alltrainx.shape[1]
    # Network parameters
    gradient_step = 1e-3
    iterations = 100
    batch_size = 25
    kernel_size = 3
    chan_input = 1
    chan_output = 32
    stride = 1
    max_pool_stride = 2
    max_pool_kernel = 2
    # loss function
    sftmax = CESoftMax()

    # Network parameters
    net = Sequential([Conv1D(kernel_size, chan_input, chan_output, stride=stride),
                      MaxPool1D(max_pool_kernel, max_pool_stride),
                      Flatten(((length - kernel_size) // stride + 1) // max_pool_stride, chan_output),
                      Linear(4064, 100),
                      ReLU(),
                      Linear(100, 10)
                      ])

    # Train networks
    opt = Optim(net=net, loss=sftmax, eps=gradient_step)
    opt.SGD(alltrainx, alltrainy_proba, batch_size, maxiter=iterations, verbose=True)
    
    predict = Softmax().forward(opt.predict(alltestx))
    y_hat = np.argmax(predict,axis=1)
    print("precision:" ,sum(y_hat == alltesty)/len(alltesty))
