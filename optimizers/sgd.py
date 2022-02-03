from . import BaseOptimizer
from utils import calculate_gradients


class SGD(BaseOptimizer):

    def __init__(self, learning_rate=0.1):
        self._learning_rate = learning_rate

    def update(self, input_data, weights, bias, loss, m_dw, m_db, v_dw, v_db):
        input_loss, dw, db = calculate_gradients(input_data, weights, loss)

        weights -= self._learning_rate * dw
        bias -= self._learning_rate * db

        return weights, bias, input_loss, m_dw, m_db, v_dw, v_db

    def callback(self):
        self._learning_rate /= 10
