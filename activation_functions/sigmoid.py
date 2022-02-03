import numpy as np

from . import BaseActivation


class Sigmoid(BaseActivation):

    def activate(self, input_data, status=''):
        return self.activate_derivative(input_data) if status == 'derivative' else 1 / (1 + np.ex(-input_data))

    def activate_derivative(self, input_data):
        return self.activate(input_data) * (1 - self.activate(input_data))
