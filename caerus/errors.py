
import numpy as np


class ErrorFunc:

    def __call__(self, yhat, y, backprop: bool=False):

        error = self._backprop(yhat, y) if backprop else self._forward(yhat, y)

        return error

    def _forward(self, yhat, y):
        pass

    def _backprop(self, yhat, y):
        pass


class MeanSquaredError(ErrorFunc):

    def _forward(self, yhat, y):
        error = y - yhat
        mse = (1 / (2 * error.shape[0])) * np.sum(np.dot(error, error.T))
        return mse

    def _backprop(self, yhat, y):
        error = (yhat - y)/y.shape[0]
        return error


class CrossEntropy(ErrorFunc):

    def _forward(self, yhat, y):
        error = np.sum(-(y * np.log2(yhat)))
        crossentropy = error / y.shape[0]
        return crossentropy

    def _backprop(self, yhat, y):
        error = (yhat - y) / y.shape[0]
        return error
