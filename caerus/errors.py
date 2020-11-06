
import numpy as np


class ErrorFunc:
    """ Base error function class. Each error function must implement a _forward and _backprop method.
    """

    def __call__(self, yhat, y, backprop: bool=False):
        """ Performs the error calculation based on predicted and target values.

        :param yhat: (numpy.ndarray). Input array of predicted value to compare against target values.
        :param y: (numpy.ndarray). Input target values to compare against predicted values.
        :param backprop: (boolean). Default is False. Flag for computing forward or backprop error calculation.
        :return: (numpy.ndarray). Resulting error computation computed between predicted and target.
        """

        error = self._backprop(yhat, y) if backprop else self._forward(yhat, y)

        return error

    def _forward(self, yhat, y):
        pass

    def _backprop(self, yhat, y):
        pass


class MeanSquaredError(ErrorFunc):
    """ Computes the mean squared error between predicted and target.
    """

    def _forward(self, yhat, y):
        """ Computes the forward pass, mean squared error between predicted and target.

        :param yhat: (numpy.ndarray). Input array of predicted values to compare against the target values.
        :param y: (numpy.ndarray). Input array of target values to compare against predicted values.
        :return: (numpy.ndarray). Resulting array of mean squared error computed between predicted and target.
        """

        error = y - yhat
        mse = (1 / (2 * error.shape[0])) * np.sum(np.dot(error, error.T))
        return mse

    def _backprop(self, yhat, y):
        """ Computes the backprop pass, mean squared error between predicted and target.

        :param yhat: (numpy.ndarray). Input array of predicted values to compare against the target values.
        :param y: (numpy.ndarray). Input array of target values to compare against predicted values.
        :return: (numpy.ndarray). Resulting array from backprop mean squared error computation between predicted and target.
        """

        error = (yhat - y)/y.shape[0]
        return error


class CrossEntropy(ErrorFunc):
    """Computes the cross entropy error between predicted and target values.
    """

    def _forward(self, yhat, y):
        """ Computes the forward pass, cross entropy error between predicted and target.

        :param yhat: (numpy.ndarray). Input array of predicted values to compare against the target values.
        :param y: (numpy.ndarray). Input array of target values to compare against predicted values.
        :return: (numpy.ndarray). Resulting array of cross entropy error computed between predicted and target.
        """
        error = np.sum(-(y * np.log(yhat + 1e-15)))
        crossentropy = error / y.shape[0]
        crossentropy = max(0, crossentropy)

        return crossentropy

    def _backprop(self, yhat, y):
        """ Computes the backprop pass, cross entropy error between predicted and target.

        :param yhat: (numpy.ndarray). Input array of predicted values to compare against the target values.
        :param y: (numpy.ndarray). Input array of target values to compare against predicted values.
        :return: (numpy.ndarray). Resulting array of cross entropy error computed between predicted and target.
        """

        error = (yhat - y) / y.shape[0]
        return error
