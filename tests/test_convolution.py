# -*- coding: utf-8 -*-

'''
Faire des tests sur les dimensions des fonctions, rapide juste un assert pour Ãªtre sur
'''
import numpy as np

from src.Activation.ReLU import ReLU
from src.Activation.softmax import Softmax
from src.Loss.CESoftMax import CESoftMax
from src.Module.conv1D import Conv1D
from src.Module.flatten import Flatten
from src.Module.linear import Linear
from src.Module.sequential import Sequential
from src.Optim.optim import Optim
from src.Pooling.maxPool1D import MaxPool1D
from src.utils.utils import load_usps, transform_numbers

if __name__ == '__main__':
    # Get the data
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)
    alltrainy_proba = transform_numbers(alltrainy, np.unique(alltrainy).shape[0])
    alltrainx = alltrainx.reshape((alltrainx.shape[0], alltrainx.shape[1], 1))
    alltestx = alltestx.reshape((alltestx.shape[0], alltestx.shape[1], 1))
    validation_size = 500
    allvalx = alltestx[:validation_size]
    allvaly = alltesty[:validation_size]
    alltestx = alltestx[validation_size:]
    alltesty = alltesty[validation_size:]

    # Get data values
    length = alltrainx.shape[1]
    # Network parameters
    gradient_step = 1e-3
    iterations = 10
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
                      Flatten(),
                      Linear(4064, 100),
                      ReLU(),
                      Linear(100, 10)
                      ])

    # Train networks
    opt = Optim(net=net, loss=sftmax, eps=gradient_step)
    opt.SGD(alltrainx, alltrainy_proba, batch_size, X_val=allvalx, Y_val=allvaly,
            f_val=lambda x: np.argmax(Softmax().forward(x), axis=1), maxiter=iterations, verbose=2)

    predict = Softmax().forward(opt.predict(alltestx))
    y_hat = np.argmax(predict, axis=1)
    print("precision:", sum(y_hat == alltesty) / len(alltesty))
