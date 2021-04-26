from src.Module.module import Module


class Flatten(Module):
    def forward(self, X):
        return X.reshape(len(X), -1)

    def backward_delta(self, input, delta):
        return delta.reshape(input.shape)
