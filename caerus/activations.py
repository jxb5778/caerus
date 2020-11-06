
import numpy as np


class Activation:
    """ Base activation class. Each activation function must implement a _forward and _backprop pass.
    """

    def __call__(self, z, backprop=False):
        """ Performs the activation operation on an input z.

        :param z: (numpy.ndarray) Input array that the activation function will be performed on.
        :param backprop: (boolean) Default is False. If true, performs the backprop operation of the activation function.
        :return: (numpy.ndarray) Resulting value from the activation operation.
        """
        activation = self._backprop(z) if backprop else self._forward(z)
        return activation

    def _forward(self, z):
        pass

    def _backprop(self, z):
        pass


class Sigmoid(Activation):
    """ Performs the sigmoid activation on an input z.
    """

    def _forward(self, z):
        """ Performs the forward pass sigmoid on an input z.

        :param z: (numpy.ndarray) Input array that the sigmoid activation will be performed on.
        :return: (numpy.ndarray) Resulting array from the sigmoid activation.
        """
        activation = 1 / (1 + np.exp(-z))
        return activation

    def _backprop(self, z):
        """ Backpropagation pass of the sigmoid activation.

        :param z: (numpy.ndarray) Input array that the backprop sigmoid will be performed.
        :return: (numpy.ndarray) Resulting array that had the backprop sigmoid performed on it.
        """
        activation = self._forward(z)
        rev_activation = activation * (1 - activation)
        return rev_activation


class Softmax(Activation):
    """ Performs the softmax activation on an input z.
    """

    def _forward(self, z):
        """ Perfoms the forward pass of the softmax activation on an input z.

        :param z: (numpy.ndarray) Input array that the softmax activation will be performed on.
        :return: (numpy.ndarray) Resulting array from the softmax activation.
        """

        exp_z = np.exp(z - np.max(z))
        activation = exp_z / np.sum(exp_z, axis=0, keepdims=True)

        return activation.T

    def _backprop(self, z):
        """ Perfoms the forward pass of the softmax activation on an input z.

        :param z: (numpy.ndarray) Input array that the softmax backprop activation will be performed on.
        :return: (int) 1.
        """
        return 1


class Relu(Activation):
    """ Performs the Relu activation function on an input z.
    """

    def _forward(self, z):
        """ Perfoms the forward pass of the relu activation on an input z.

        :param z: (numpy.ndarray) Input array that the relu activation will be performed on.
        :return: (numpy.ndarray) Resulting array from the relu activation.
        """
        activation = z * (z > 0)
        return activation

    def _backprop(self, z):
        """ Perfoms the forward pass of the softmax activation on an input z.

        :param z: (numpy.ndarray) Input array that the relu activation will be performed on.
        :return: (numpy.ndarray) Resulting array from the relu activation.
        """
        rev_activation = 1.0 * (z > 0)
        return rev_activation
