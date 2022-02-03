import abc


class BaseOptimizer(abc.ABC):

    @abc.abstractmethod
    def update(self, input_data, weights, bias, loss, m_dw, m_db, v_dw, v_db):
        pass

    @abc.abstractmethod
    def callback(self):
        pass
