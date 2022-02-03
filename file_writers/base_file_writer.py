import abc


class BaseFileWriter(abc.ABC):

    def __init__(self):
        self._epochs = {}
        self._optimizer = ''
        self._loss_func = ''
        self._accuracy = 0

    @abc.abstractmethod
    def set_values(self, epochs, optimizer, loss_func, accuracy):
        pass

    @abc.abstractmethod
    def enroll(self, epochs, optimizer, loss_func, accuracy):
        pass

    @abc.abstractmethod
    def create_file(self, fname, epochs, optimizer, loss_func, accuracy):
        pass
