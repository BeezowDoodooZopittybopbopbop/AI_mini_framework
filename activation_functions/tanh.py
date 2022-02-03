import numpy as np

from . import BaseActivation


class Tanh(BaseActivation):

    def activate(self, input_data, status=''):
        return self.activate_derivative(input_data) if status == 'derivative' else np.tanh(input_data)

    def activate_derivative(self, input_data):
        return 1 - np.tanh(input_data) ** 2
