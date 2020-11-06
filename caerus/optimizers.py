
import numpy as np


class SGD:
    """ Performs stochastic gradient decent with momentum.

    :param learning_rate: (float). Default is 0.001. Learning rate value to control gradient step size during training.
    :param beta_1: (float). Default is 0.9. Beta value to use to control gradient momentum during training.
    :param grad_clip: (int). Default is None. If weights result above or below this value, they will be capped to this value.

    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, grad_clip: int=None):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1

        self.grad_clip = grad_clip

        # :param _update_list: (list). Contains the weight updates for the previous update.
        self._update_list = None

    def __call__(self, delta, outs):
        """ Calculates the weight update value for a caerus.layers.Dense layer.

        :param delta: (numpy.ndarray). Input delta array to cross product with the output value.
        :param outs: (numpy.ndarray). Input activation array to cross product with the delta value.
        :return: (numpy.ndarray). Resulting array from the cross product of the delta and activation times the learning rate.
        """

        delta_w = self.learning_rate * np.dot(delta, outs)

        return delta_w

    def momentum(self, weight_list, update_list):
        """ Computes the momentum updates for the weights.

        :param weight_list: (numpy.ndarray). Input weight list to incorporate momentum update.
        :param update_list: (numpy.ndarray). Input weight update list to store for next iteration of momentum updates.
        :return: (numpy.ndarray). Resulting weights with momentum update incorporated.
        """

        if self._update_list is None:
            self._update_list = update_list
            return weight_list

        update_w_momentum = [
            weight + self.beta_1 * prev_update
            for weight, prev_update in zip(weight_list, self._update_list)
        ]

        if self.grad_clip is not None:
            update_w_momentum = [
                np.clip(update, a_min=-self.grad_clip, a_max=self.grad_clip) for update in update_w_momentum
            ]

        self._update_list = update_list

        return update_w_momentum
