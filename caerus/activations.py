
import numpy as np


class Activation:
    def __call__(self, z, backprop=False):
        activation = self._backprop(z) if backprop else self._forward(z)
        return activation

    def _forward(self, z):
        pass

    def _backprop(self, z):
        pass


class Sigmoid(Activation):

    def _forward(self, z):
        activation = 1 / (1 + np.exp(-z))
        return activation

    def _backprop(self, z):
        activation = self._forward(z)
        rev_activation = activation * (1 - activation)
        return rev_activation


class Softmax(Activation):

    def _forward(self, z):
        exp_z = np.exp(z - np.max(z))
        activation = exp_z / np.sum(exp_z, axis=0, keepdims=True)

        return activation.T

    def _backprop(self, z):

        return 1
"""
        activation = self._forward(z)

        rev_activation = np.array([
            np.diagflat(dist.reshape(-1, 1)) - np.dot(dist.reshape(-1, 1), dist.reshape(-1, 1).T)
            for dist in activation
        ])

        rev_activation = (1/rev_activation.shape[0]) * np.sum(rev_activation, axis=0)

        return rev_activation
"""

class Relu(Activation):

    def _forward(self, z):
        activation = z * (z > 0)
        return activation

    def _backprop(self, z):
        rev_activation = 1.0 * (z > 0)
        return rev_activation
