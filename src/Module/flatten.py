from src.Module.module import Module


class Flatten(Module):
    def __init__(self, length, chan_in):
        self._chan_in = chan_in
        self._length = length

    def forward(self, X):
        p = X
        p.flatten()
        return p

    def backward_delta(self, input, delta):
        z = delta
        z.reshape(self._length, self._chan_in)
        return z
