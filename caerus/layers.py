
import numpy as np

from caerus import activations


class Layer:
    """ Base layer class. Each Layer must implement a method for init_weights, forward, and backprop.
    """

    def init_weights(self, output_units: int):
        pass

    def _validate_activation(self, activation: str):
        """ Wrapper to allow string inputs when setting activation in Layers.

        :param activation: (str or caerus.activations.Activation). Input activation to validate.
        :return: (caerus.activations.Activation). Resulting validated activation function to use in the Layer.
        """

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
    """ Input layer of a MLP, defines the size of the input to the network.

    :param input_shape: (tuple of int). Tuple of values describing the shape of the input data to the network.
    :param name: (str). Default is 'Input'. Name to use for this layer of the network.

    """

    def __init__(self, input_shape: tuple, name: str = 'Input'):

        self.input_shape = input_shape
        self.units = np.prod(input_shape)

        self.out = None
        self.z = None

        self.size = input_shape
        self.name = name

    def activation(self, z, backprop: bool=True):
        """ Function to use as the activation of the input layer- Returns 1.

        :param z: (numpy.ndarray). Input array of z values to compute activation.
        :param backprop: (boolean). Flag for weather to compute forward or backward pass.
        :return: (int). Returns 1 as the activation.
        """
        return 1

    def forward(self, X):
        """ Computes the forward pass for the input, storing the input array as output and z.

        :param X: (numpy.ndarray). Input array of values to pass through the network.
        :return: (numpy.ndarray). Returns the input array of values to pass through the network.
        """

        self.out = X
        self.z = X
        return X


class Dense(Layer):
    """ Fully connected layer.

    :param units: (int). Number of output units from this layer in the network.
    :param activation: (str or caerus.activation.Activation). Activation function to use for this layer.
    :param name: (str). Default is 'Dense'. Name to use for this layer in the network.

    """

    def __init__(self, units: int, activation: str, name: str = 'Dense'):

        self.units = units
        self.activation = self._validate_activation(activation)

        self.size = None
        self.name = name

        self.weights = None
        self.biases = None

        self.z = None
        self.out = None

    def init_weights(self, input_units: int):
        """ Initializes the weights and the biases for this layer.

        :param input_units: (int). Number of input units that connect to this layer.
        :return: None. The weights will be randomly set with size (self.units, @input_units), and biases (self.units, 1).
        """

        self.weights = np.random.rand(self.units, input_units)
        self.biases = np.random.rand(self.units, 1)

        self.size = self.weights.shape

    def forward(self, X):
        """ Forward pass of the full connect layer- computes z term and returns the activation of z.

        Notes:
            Updates internal parameters z and out, and returns out.

        :param X: (numpy.ndarray). Input array for computing the z and activation.
        :return: (numpy.ndarray). Resulting array after computing z, the dot product between
            weights and @X plus the bias, and computing the activation on the z term.
        """

        self.z = np.dot(self.weights, X) + self.biases
        self.out = self.activation(self.z)

        return self.out

    def backprop(self, weight_update, bias_update):
        """ Backpropagation pass of the Dense layer. Updates the layer weights and biases.

        :param weight_update: (numpy.ndarray). Input array for how much to update the weights.
        :param bias_update: Inpit array for how much to update the biases.
        :return: None. Updates internal parameters weights and biases.
        """

        self.weights -= weight_update
        self.biases -= bias_update
