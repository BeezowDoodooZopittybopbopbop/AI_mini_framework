import abc


class BaseLayer(abc.ABC):

    def __init__(self):
        self._input = None
        self._output = None

    @abc.abstractmethod
    def forward(self, input_data):
        pass

    @abc.abstractmethod
    def backward(self, loss, optimizer):
        pass
