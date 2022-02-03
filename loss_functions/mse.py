import numpy as np

from . import BaseLoss


class MSE(BaseLoss):

    def calculate(self, y_true, y_pred, status=''):
        return self.calculate_derivative(y_true, y_pred) if status == 'derivative' else np.mean(
            np.power(y_true - y_pred, 2))

    def calculate_derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
