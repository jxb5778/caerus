
import numpy as np

from caerus import activations


class Layer:

    def init_weights(self, output_units: int):
        pass

    def _validate_activation(self, activation: str):

        validated_activation = activation

        if isinstance(activation, str):
            if activation == 'sigmoid':
                validated_activation = activations.Sigmoid()

            elif activation == 'relu':
                validated_activation = activations.Relu()

            elif activation == 'softmax':
                validated_activation = activations.Softmax()

            else:
                message = """
                You must specify an activation of : 'sigmoid', 'softmax', or 'relu'
                """
                raise ValueError(message)

        elif isinstance(activation, activations.Activation):
            validated_activation = activation

        else:
            message = """
            You must specify an activation of type string or caerus.activation.Activation.
            """
            raise ValueError(message)

        return validated_activation

    def forward(self, X):
        pass

    def backprop(self, **kwargs):
        pass


class Input(Layer):

    def __init__(self, input_shape: tuple, name: str='Input'):

        self.input_shape = input_shape
        self.units = np.prod(input_shape)

        self.out = None
        self.z = None

        self.size = input_shape
        self.name = name

    def activation(self, z, backprop: bool=True):
        return 1

    def forward(self, X):
        self.out = X
        self.z = X
        return X


class Dense(Layer):

    def __init__(self, units: int, activation: str, name: str='Dense'):

        self.units = units
        self.activation = self._validate_activation(activation)

        self.size = None
        self.name = name

        self.weights = None
        self.biases = None

        self.z = None
        self.out = None

    def init_weights(self, input_units: int):
        self.weights = np.random.rand(self.units, input_units)
        self.biases = np.random.rand(self.units, 1)

        self.size = self.weights.shape

    def forward(self, X):
        self.z = np.dot(self.weights, X) + self.biases
        self.out = self.activation(self.z)

        return self.out

    def backprop(self, weight_update, bias_update):

        self.weights -= weight_update
        self.biases -= bias_update
