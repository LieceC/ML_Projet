# -*- coding: utf-8 -*-

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, adjusted_rand_score

from src.Activation.sigmoid import Sigmoid
from src.Activation.tanH import TanH
from src.Loss.BCE import BCE
from src.Module.Linear import Linear
from src.Module.sequential import Sequential
from src.Optim.Optim import Optim
from src.utils.utils import load_usps


def TNSE(alltesty, compression):
    """
    alltesty : y des données de test
    compression : compression des données de test
    """
    # apply TSNE on data
    tsne = TSNE()
    colors = ["red", "green", "blue", "yellow", "brown", "orange", "black", "cyan", "violet", "pink"]
    classes = tsne.fit_transform(compression)
    patches = [mpatches.Patch(color=colors[i], label=str(i)) for i in range(len(colors))]
    plt.legend(handles=patches)
    for y in np.unique(alltesty):
        classes_y = classes[np.where(alltesty == y)]

        plt.scatter(classes_y[:, 0], classes_y[:, 1], c=colors[y])
    plt.show()


def cluster(comp_train, comp_test, alltrainy, alltesty):
    # clustering
    clustering = KMeans(10, max_iter=1000)
    clusters_train = clustering.fit_predict(comp_train, alltrainy)
    clusters_test = clustering.predict(comp_test)

    conf = confusion_matrix(alltesty, clusters_test)
    plt.imshow(conf)
    plt.show()

    maximums = np.argmax(conf, axis=1)
    only_argmax = np.zeros((10, 10))
    for x in range(10):
        only_argmax[x, maximums[x]] = conf[x, maximums[x]]

    plt.imshow(only_argmax)
    plt.show()

    puretes = []
    for x in range(10):
        puretes += [conf[x, maximums[x]] / sum(conf[x])]
        print("cluster", x, " - pureté:", puretes[-1])

    print("pureté moyenne:", np.array(puretes).mean())

    print("Rand Score :", adjusted_rand_score(alltesty, clusters_test))
    return clusters_test


def test_auto_encodeur():
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)
    alltrainx /= 2
    alltestx /= 2

    # Initialize modules with respective size
    iteration = 100
    gradient_step = 1e-3
    batch_size = 1000  # len(alltrainx)

    input_size = alltrainx.shape[1]
    hidden_size = 100
    compression_size = 10

    m_linear = Linear(input_size, hidden_size)
    m_tanh = TanH()
    m_linear2 = Linear(hidden_size, compression_size)

    m_linear3 = Linear(compression_size, hidden_size)
    m_linear3._parameters = m_linear2._parameters.T

    m_linear4 = Linear(hidden_size, input_size)
    m_linear4._parameters = m_linear._parameters.T

    m_sigmoid = Sigmoid()

    m_loss = BCE()

    seq = Sequential([m_linear, m_tanh, m_linear2, m_tanh, m_linear3, m_tanh, m_linear4, m_sigmoid])

    opt = Optim(seq, loss=m_loss, eps=gradient_step)
    opt.SGD(alltrainx, alltrainx, batch_size, maxiter=iteration, verbose=True)

    # predict = opt.predict(alltestx)

    '''
    # print 
    for i in range(6):
        plt.imshow(alltestx[i].reshape((16,16)))
        plt.show()
        plt.imshow(predict[i].reshape((16,16)))
        plt.show()
    '''
    compression_train = seq.forward(alltrainx)[-5]
    compression = seq.forward(alltestx)[-5]

    # TNSE(alltesty,compression)
    return cluster(compression_train, compression, alltrainy, alltesty)


if __name__ == '__main__':
    test = test_auto_encodeur()
