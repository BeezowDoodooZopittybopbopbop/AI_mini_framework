import json

from . import BaseFileWriter


class Json(BaseFileWriter):

    def __init__(self):
        super().__init__()
        self._json = {}

    def set_values(self, epochs, optimizer, loss_func, accuracy):
        self._epochs = epochs
        self._optimizer = optimizer
        self._loss_func = loss_func
        self._accuracy = accuracy

    def enroll(self, epochs, optimizer, loss_func, accuracy):
        self.set_values(epochs, optimizer, loss_func, accuracy)
        self._json['Training'] = self._epochs
        self._json['Optimizer'] = self._optimizer
        self._json['Loss Function'] = self._loss_func
        self._json['Accuracy'] = self._accuracy

    def create_file(self, fname, epochs, optimizer, loss_func, accuracy):
        self.enroll(epochs, optimizer, loss_func, accuracy)
        json_object = json.dumps(self._json, indent=4)

        with open(fname, "w") as outfile:
            outfile.write(json_object)
