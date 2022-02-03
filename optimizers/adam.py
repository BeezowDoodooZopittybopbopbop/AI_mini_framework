import numpy as np

from . import BaseOptimizer
from utils import calculate_gradients


class Adam(BaseOptimizer):

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def update(self, input_data, weights, bias, loss, m_dw, m_db, v_dw, v_db):
        input_loss, dw, db = calculate_gradients(input_data, weights, loss)

        m_dw = self._beta1 * m_dw + (1 - self._beta1) * dw
        m_db = self._beta1 * m_db + (1 - self._beta1) * db

        v_dw = self._beta2 * v_dw + (1 - self._beta2) * (dw ** 2)
        v_db = self._beta2 * v_db + (1 - self._beta2) * (db ** 2)

        m_dw_corr = m_dw / (1 - self._beta1)
        m_db_corr = m_db / (1 - self._beta1)
        v_dw_corr = v_dw / (1 - self._beta2)
        v_db_corr = v_db / (1 - self._beta2)

        weights -= self._learning_rate * (m_dw_corr / (np.sqrt(v_dw_corr) + self._epsilon))
        bias -= self._learning_rate * (m_db_corr / (np.sqrt(v_db_corr) + self._epsilon))

        return weights, bias, input_loss, m_dw, m_db, v_dw, v_db

    def callback(self):
        self._learning_rate /= 10
