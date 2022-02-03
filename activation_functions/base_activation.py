import abc


class BaseActivation(abc.ABC):

    @abc.abstractmethod
    def activate(self, input_data, status=''):
        pass

    @abc.abstractmethod
    def activate_derivative(self, input_data):
        pass
