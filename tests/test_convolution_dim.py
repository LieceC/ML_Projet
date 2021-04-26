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
    batch_size = 25
    kernel_size = 3
    chan_input = 1
    chan_output = 32
    stride = 2
    length = 128
    
    """
        Convolution
    """
    X = np.random.rand(batch_size,length,chan_input)
    convolution = Conv1D(kernel_size, chan_input, chan_output, stride=stride)
    
    # forward size
    forward_conv = convolution.forward(X)
    assert forward_conv.shape == (batch_size, (length-kernel_size)//stride +1,chan_output)
    
    i = 2 # pixel de test
    image = 0 # image de test
    filtre = 1 # filtre de test
    
    # operateur forward result
    res = convolution.operateur(X,i*stride)[image][filtre]
    w = convolution._parameters[:,:,filtre]
    excepted = np.sum(w*X[image,i*stride:i*stride+kernel_size])   
    assert np.abs(excepted - res) < 1e-5
    
    # forward result
    res = forward_conv[image][i][filtre]
    assert np.abs(res - excepted) < 1e-5
    
    
    # backward size
    delta = np.random.rand(forward_conv.shape[0],forward_conv.shape[1],forward_conv.shape[2])
    backward_conv = convolution.backward_delta(X, delta)
    assert backward_conv.shape == (batch_size,length,chan_input)
    
    """
    # backward_delta result
    res = convolution.backward_delta(X,delta)#[image][filtre]
    print(res.shape)
    excepted = np.sum(convolution._parameters[:,:,filtre])
    assert np.abs(excepted - res) < 1e-5
    """
