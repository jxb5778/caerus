
import numpy as np

from caerus import activations
from caerus import errors


class Model:
    """ Base model class, each model must implement a fit and a predict method.

    """

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass

    def compile(self, loss, optimizer):
        self.loss_func = loss
        self.optimizer = optimizer
        self._is_compiled = True


class MLP(Model):
    """ Base multi-layer perceptron model (binary classification)

    Note:
        This model wasn't used for project milestone 2 results -> please review Sequential.

    """

    def __init__(self, layers: list, rand_seed: int=42):
        self.weight_list = list()
        self.bias_list = list()

        self._is_compiled = False

        np.random.seed(rand_seed)

        for index, units in enumerate(layers):
            if index == len(layers) - 1:
                break

            weights = np.random.rand(layers[index+1], units)
            bias = np.random.rand(layers[index+1], 1)

            self.weight_list.append(weights)
            self.bias_list.append(bias)

    def summary(self):

        model_summary = """Model Summary:\n============================================\n{}     
        """.format(
            '\n'.join(
                "Layer " + str(index+1)
                + "| Weights: " + str(weight_bias[0].shape)
                + "; Biases: " + str(weight_bias[1].shape)

            for index, weight_bias in enumerate(zip(self.weight_list, self.bias_list))
            )
        )

        return model_summary

    def _forward(self, X):

        activation = activations.Sigmoid()

        self.z_list = list()
        self.out_list = list()

        output = X.T
        self.out_list.append(output)

        for index, weight_bias in enumerate(zip(self.weight_list, self.bias_list)):

            weight, bias = weight_bias

            z = np.dot(weight, output) + bias
            output = activation(z)

            self.z_list.append(z)
            self.out_list.append(output)

        return output

    def _backprop(self, yhat_batch, y_batch):

        batch_size = y_batch.shape[0]

        activation = activations.Sigmoid()

        update_list = list()

        delta = self.loss_func(yhat_batch, y_batch, backprop=True) * activation(self.z_list[-1], backprop=True)

        weight_update = (1/batch_size) * self.optimizer(delta, self.out_list[-1].T)
        update_list.append(weight_update)

        self.weight_list[-1] -= weight_update
        self.bias_list[-1] -= (1/batch_size) * self.optimizer.learning_rate * np.sum(delta, axis=1, keepdims=True)

        for layer in range(2, len(self.weight_list) + 1):

            z = self.z_list[-layer]

            delta = np.dot(self.weight_list[-layer+1].T, delta) * activation(z, backprop=True)

            weight_update = self.optimizer(delta, self.out_list[-layer-1].T)
            update_list.append(weight_update)

            self.weight_list[-layer] -= weight_update
            self.bias_list[-layer] -= (1/batch_size) * self.optimizer.learning_rate * np.sum(delta, axis=1, keepdims=True)

        self.weight_list = self.optimizer.momentum(self.weight_list, update_list[::-1])

    def fit(self, X, y=None, epochs: int=10, batch_size: int=1, rand_seed: int=42):

        if not self._is_compiled:
            message = """
            You must compile your model with a loss function and optimizer before you can run the .fit method.
            """
            raise ValueError(message)

        X = X.to_numpy()
        y = y.to_numpy()

        history = list()

        np.random.seed(rand_seed)

        for _ in range(epochs):
            np.random.shuffle(X)

            X_batch_list = np.array_split(X, X.shape[0]/batch_size)
            y_batch_list = np.array_split(y, y.shape[0]/batch_size)

            for X_batch, y_batch in zip(X_batch_list, y_batch_list):

                yhat_batch = self._forward(X_batch)

                error = self.loss_func(yhat_batch, y_batch)

                self._backprop(yhat_batch, y_batch)

            epoch_error = self.loss_func(self._forward(X), y)
            history.append(epoch_error)

        return history

    def predict(self, X):
        X = X.to_numpy()
        y_hat = self._forward(X)

        return y_hat


class Sequential(Model):
    """ Sequential feed-forward neural network model.

    :param layers: (list of caerus.layers.Layer). List of layers to compute sequentially during forward and backprop passes.
    :param rand_seed: (int). Default is 42. Integer value to use as the random seed.
        Utilized during shuffling the input data after each epoch, as well as weight and bias initializations.
    """

    def __init__(self, layers: list, rand_seed: int=42):

        self.layers = layers
        self._is_compiled = False

        np.random.seed(rand_seed)

    def compile(self, loss, optimizer):
        """ Sets the loss function and optimizer for training of the model.

        :param loss: (str or caerus.errors.ErrorFunc). Function to use to compute loss during training.
            String inputs can be either: 'mse' or 'crossentropy'.
        :param optimizer: (caerus.optimizers.SGD). Optimizer to use during training to optimize the model.
        :return: None. sets model internal parameters: loss_func, optimizer, and sets _is_compiled to True.
        """

        if isinstance(loss, str):

            if loss == 'mse':
                self.loss_func = errors.MeanSquaredError()

            elif loss == 'crossentropy':
                self.loss_func = errors.CrossEntropy()

            else:
                message = """
                You must specify a loss function of : 'mse', 'crossentropy', or ...
                """
                raise ValueError

        else:
            self.loss_func = loss

        self.optimizer = optimizer
        self._is_compiled = True

        for idx in range(1, len(self.layers)):
            self.layers[idx].init_weights(input_units=self.layers[idx - 1].units)
            self.layers[idx].name = '{}_{}'.format(self.layers[idx].name, idx)

    def summary(self):
        """ Returns a string representation for the summary of the model.

        :return: (str). Resulting string representation summarizing the model architecture.
        """

        model_summary = """Model Summary:\n============================================\n{}\n""".format(
            '\n'.join(layer.name + ": " + str(layer.size) for layer in self.layers)
        )

        return model_summary

    def _forward(self, X):
        """ Sequentially computes the forward pass through the model.

        :param X: (numpy.ndarray). Input array to compute the forward pass.
        :return: (numpy.ndarray). Resulting array after computing the forward pass through the network.
        """

        output = X.T

        for index, layer in enumerate(self.layers):
            output = layer.forward(output)

        return output

    def _backprop(self, yhat_batch, y_batch):
        """ Sequentially backpropogates loss through the network.

        :param yhat_batch: (numpy.ndarray). Resulting target predictions after forward pass through the network.
        :param y_batch: (numpy.ndarray). Targets for input data.
        :return: None. Internal weights and bias parameters will be updated after error is backpropagated.
        """

        batch_size = y_batch.shape[0]

        update_list = list()

        delta = self.loss_func(yhat_batch, y_batch, backprop=True) * self.layers[-1].activation(self.layers[-1].z, backprop=True)

        weight_update = (1 / batch_size) * self.optimizer(delta, self.layers[-1].out.T)

        update_list.append(weight_update)

        bias_update = (1 / batch_size) * self.optimizer.learning_rate * np.sum(delta, axis=1, keepdims=True)

        self.layers[-1].backprop(weight_update, bias_update)

        delta = delta.T

        for l_idx in range(2, len(self.layers)):
            layer = self.layers[-l_idx]

            delta = np.dot(self.layers[-l_idx + 1].weights.T, delta) * layer.activation(layer.z, backprop=True)

            weight_update = (1 / batch_size) * self.optimizer(delta, self.layers[-l_idx - 1].out.T)
            update_list.append(weight_update)

            bias_update = (1 / batch_size) * self.optimizer.learning_rate * np.sum(delta, axis=1, keepdims=True)

            self.layers[-l_idx].backprop(weight_update, bias_update)

        weight_list = [layer.weights for layer in self.layers[1:]]
        weight_w_momentum_list = self.optimizer.momentum(weight_list, update_list[::-1])

        for l_idx in range(len(self.layers) - 1):
            self.layers[l_idx+1].weights = weight_w_momentum_list[l_idx]

    def fit(self, X, y=None, epochs: int = 10, batch_size: int = 1, rand_seed: int = 42):
        """ Performs tuning of the model based on input data X and targets y.

        :param X: (numpy.ndarray). Input data to use for model training.
        :param y: (numpy.ndarray). Input data to use as the target during model training.
        :param epochs: (int). Number of epochs to train the model.
        :param batch_size: (int). Number of data points to use in each batch of training data.
        :param rand_seed: (int). Random seed to use during model training- affects shuffle of data between epochs.
        :return: (list of float). Resulting history of errors during model training after each epoch.
        """

        if not self._is_compiled:
            message = """
            You must compile your model with a loss function and optimizer before you can run the .fit method.
            """
            raise ValueError(message)

        if not isinstance(X, np.ndarray):
            X = X.to_numpy()

        if not isinstance(y, np.ndarray):
            y = y.to_numpy()

        np.random.seed(rand_seed)
        history = list()

        for _ in range(epochs):
            shuffle_idx = np.random.permutation(X.shape[0])
            X = X[shuffle_idx]
            y = y[shuffle_idx]

            X_batch_list = np.array_split(X, X.shape[0]/batch_size)
            y_batch_list = np.array_split(y, y.shape[0]/batch_size)

            for X_batch, y_batch in zip(X_batch_list, y_batch_list):

                yhat_batch = self._forward(X_batch)

                error = self.loss_func(yhat_batch, y_batch)
                print("Error: ", error)
                print("Yhat: ", yhat_batch)
                print("Y: ", y_batch)
                print()

                self._backprop(yhat_batch, y_batch)

            epoch_error = self.loss_func(self._forward(X), y)
            history.append(epoch_error)

        return history

    def predict(self, X):
        """ Performs inference on input data using the trained model.

        :param X: (numpy.ndarray). Input data that will have target labels inferred.
        :return: (numpy.ndarray). Resulting targets from inference, using the trained model.
        """

        if not isinstance(X, np.ndarray):
            X = X.to_numpy()

        y_hat = self._forward(X)

        print("Predicted: ", y_hat)

        return y_hat
