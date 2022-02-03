import abc


class BaseLoss(abc.ABC):

    @abc.abstractmethod
    def calculate(self, y_true, y_pred, status=''):
        pass

    @abc.abstractmethod
    def calculate_derivative(self, y_true, y_pred):
        pass
