from src.Module.module import Module


class Flatten(Module):
    def __init__(self, length, chan_in):
        self._chan_in = chan_in
        self._length = length

    def forward(self, X):
        p = X
        p.flatten(axis=1)
        return p

    def backward_delta(self, input, delta):
        z = delta
        z.reshape(delta.shape[0], self._length, self._chan_in)
        return z
