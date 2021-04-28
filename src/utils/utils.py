import matplotlib.pyplot as plt
import numpy as np

from src.utils.mltools import make_grid


def plot_frontiere_proba(data, f, step=20):
    grid, x, y = make_grid(data=data, step=step)
    plt.contourf(x, y, f(grid).reshape(x.shape), 255)
    plt.show()


def show_usps(data):
    plt.figure()
    plt.imshow(data.reshape((16, 16)), interpolation="nearest", cmap="gray")
    plt.show()


def load_usps(fn):
    with open(fn, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    return tmp[:, 1:], tmp[:, 0].astype(int)


def get_usps(l, datax, datay):
    if type(l) != list:
        resx = datax[datay == l, :]
        resy = datay[datay == l]
        return resx, resy
    tmp = list(zip(*[get_usps(i, datax, datay) for i in l]))
    return np.vstack(tmp[0]), np.hstack(tmp[1])


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def transform_numbers(input, size):
    """Assume 1D array as input, len is the number of example
    Transform into proba
    """
    datay_r = np.zeros((len(input), size))
    # Re-arranging data to compute a probability
    datay_r[np.arange(len(input)), input] = 1
    return datay_r
