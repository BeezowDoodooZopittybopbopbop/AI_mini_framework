from . import BaseLayer


class Activation(BaseLayer):

    def __init__(self, function):
        super().__init__()
        self._function = function

    def forward(self, input_data):
        self._input = input_data
        self._output = self._function.activate(self._input)

        return self._output

    def backward(self, loss, optimizer):
        return self._function.activate(self._input, 'derivative') * loss
