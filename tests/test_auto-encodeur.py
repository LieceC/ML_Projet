# -*- coding: utf-8 -*-

import numpy as np
import sklearn.metrics as skt

from src.Activation.sigmoid import Sigmoid
from src.Activation.tanH import TanH
from src.Loss.BCE import BCE
from src.Loss.MSELoss import MSELoss
from src.Module.Linear import Linear
from src.utils.utils import load_usps
from src.Module.sequential import Sequential
from src.Optim.Optim import Optim
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

def test_auto_encodeur():
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)
    alltrainx/=2
    alltestx/=2
    
    # Initialize modules with respective size
    iteration = 100
    gradient_step = 1e-3
    batch_size = 10 # len(alltrainx)
    
    input_size  = alltrainx.shape[1]
    hidden_size = 100
    nb_class = 10
   
    m_linear = Linear(input_size, hidden_size)
    m_tanh = TanH()
    m_linear2 = Linear(hidden_size, nb_class)
    
    m_linear3 = Linear(nb_class, hidden_size)
    m_linear3._parameters = m_linear2._parameters.T
    
    m_linear4 = Linear(hidden_size, input_size)
    m_linear4._parameters = m_linear._parameters.T
    
    m_sigmoid = Sigmoid()
    
    m_loss = BCE()
    
    seq = Sequential([m_linear,m_tanh,m_linear2,m_tanh,m_linear3,m_tanh,m_linear4,m_sigmoid])
    
    opt = Optim(seq,loss=m_loss,eps = gradient_step)
    opt.SGD(alltrainx,alltrainx,batch_size, maxiter=iteration,verbose = True)

    predict = opt.predict(alltestx)
    
    for i in range(6):
        plt.imshow(alltestx[i].reshape((16,16)))
        plt.show()
        plt.imshow(predict[i].reshape((16,16)))
        plt.show()
        
    classes = seq.forward(alltestx)[-5]
    tsne = TSNE()
    colors = ["red","green","blue","yellow","brown","orange","black","cyan","violet","pink"]
    classes = tsne.fit_transform(classes)
    patches = [mpatches.Patch(color=colors[i],label=str(i)) for i in range(len(colors))]
    plt.legend(handles=patches)
    for y in np.unique(alltesty):
        classes_y = classes[np.where(alltesty==y)]
        
        plt.scatter(classes_y[:,0],classes_y[:,1],c=colors[y])
    plt.plot()
    return predict
    # predict = np.argmax(predict, axis=1)

    # res = skt.confusion_matrix(predict, alltesty)
    # print(np.sum(np.where(predict==alltesty,1,0))/len(predict))
    # plt.imshow(res)
    
    

if __name__ == '__main__':
    test = test_auto_encodeur()
    