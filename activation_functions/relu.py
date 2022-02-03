import numpy as np

from . import BaseActivation


class ReLU(BaseActivation):

    def activate(self, input_data, status=''):
        if status == 'derivative':
            return self.activate_derivative(input_data)
        data = [max(0.05 * value, value) for array in input_data for value in array]
        return np.array(data).reshape(input_data.shape)

    def activate_derivative(self, input_data):
        data = [1 if value > 0 else 0.05 for array in input_data for value in array]
        return np.array(data).reshape(input_data.shape)
