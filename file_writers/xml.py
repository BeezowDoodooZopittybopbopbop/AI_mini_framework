from xml.dom import minidom

from .base_file_writer import BaseFileWriter


class XML(BaseFileWriter):

    def __init__(self):
        super().__init__()
        self._root = minidom.Document()
        self._xml_results = self._root.createElement('results')
        self._root.appendChild(self._xml_results)

    def xml_element(self, name_element, name_info, info):
        xml_element = self._root.createElement(name_element)
        xml_element.setAttribute(name_info, info)
        return xml_element

    def set_values(self, epochs, optimizer, loss_func, accuracy):
        self._epochs = epochs
        self._optimizer = optimizer
        self._loss_func = loss_func
        self._accuracy = accuracy

    def enroll(self, epochs, optimizer, loss_func, accuracy):
        self.set_values(epochs, optimizer, loss_func, accuracy)
        for e, l in self._epochs.items():
            self._xml_results.appendChild(self.xml_element(e, 'loss', str(l)))
        self._xml_results.appendChild(self.xml_element('optimizer', 'name', type(self._optimizer).__name__))
        self._xml_results.appendChild(self.xml_element('loss function', 'name', type(self._loss_func).__name__))
        self._xml_results.appendChild(self.xml_element('accuracy', 'value', str(self._accuracy)))

    def create_file(self, fname, epochs, optimizer, loss_func, accuracy):
        self.enroll(epochs, optimizer, loss_func, accuracy)
        xml_str = self._root.toprettyxml(indent="\t")

        with open(fname, "w") as outfile:
            outfile.write(xml_str)
