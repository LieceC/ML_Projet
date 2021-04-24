from src.Module.module import Module


class Flatten(Module):
    def __init__(self, length, chan_in):
        super().__init__()
        self._chan_in = chan_in
        self._length = length

    def forward(self, X):
        p = X
        p = p.reshape(X.shape[0], X.shape[1] * X.shape[2])
        return p

    def backward_delta(self, input, delta):
        return delta.reshape(delta.shape[0], self._length, self._chan_in)
