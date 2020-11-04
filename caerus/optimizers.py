
import numpy as np


class SGD:

    def __init__(self, learning_rate=0.001, beta_1=0.9, grad_clip: int=None):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1

        self.grad_clip = grad_clip

        self._update_list = None

    def __call__(self, delta, outs):

        delta_w = self.learning_rate * np.dot(delta, outs)

        return delta_w

    def momentum(self, weight_list, update_list):

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
