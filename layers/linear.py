import numpy as np

from . import BaseLayer


class Linear(BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self._weights = np.random.rand(input_size, output_size) - 0.5
        self._bias = np.random.rand(1, output_size) - 0.5
        self._m_dw = np.zeros((input_size, output_size))
        self._m_db = np.zeros((1, output_size))
        self._v_dw = np.zeros((input_size, output_size))
        self._v_db = np.zeros((1, output_size))

    def forward(self, input_data):
        self._input = input_data
        self._output = np.dot(self._input, self._weights) + self._bias

        return self._output

    def backward(self, loss, optimizer):
        self._weights, self._bias, input_loss, self._m_dw, self._m_db, self._v_dw, self._v_db = optimizer.update(
            self._input, self._weights, self._bias, loss,
            self._m_dw, self._m_db, self._v_dw, self._v_db)

        return input_loss
