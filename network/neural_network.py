from utils import get_value


class NeuralNetwork:

    def __init__(self):
        self._layers = []
        self._epochs = {}
        self._optimizer = ''
        self._loss_func = ''
        self._accuracy = 0

    def add(self, layer):
        self._layers.append(layer)

    def fit(self, x_train, y_train, epochs, optimizer, loss_func, callback=False):
        for epoch in range(epochs):
            if (epoch % 30) == 0 and callback is True:
                optimizer.callback()

            loss_display = 0
            for i in range(len(x_train)):
                output = x_train[i]

                # forward propagation
                for layer in self._layers:
                    output = layer.forward(output)

                # calculating errors (1st to display, 2nd for backpropagation)
                loss_display += loss_func.calculate(y_train[i], output)
                loss = loss_func.calculate(y_train[i], output, 'derivative')

                # backward propagation
                for layer in reversed(self._layers):
                    loss = layer.backward(loss, optimizer)

            loss_epoch = loss_display / len(x_train)
            print('Epoch: {}/{}  Loss: {}'.format(epoch + 1, epochs, loss_epoch))

            self._epochs['Epoch: {}/{}'.format(epoch + 1, epochs)] = loss_epoch

        self._optimizer = type(optimizer).__name__
        self._loss_func = type(loss_func).__name__

    def predict(self, input_data, true_value):
        predictions = []

        # run network over all samples
        for i in range(len(input_data)):
            # forward propagation
            output = input_data[i]
            for layer in self._layers:
                output = layer.forward(output)

            output = output.flatten()
            predictions.append(get_value(output) == get_value(true_value[i]))

        accuracy = (sum(predictions) / len(predictions)) * 100
        print('Accuracy: {}'.format(accuracy))
        self._accuracy = accuracy

        return accuracy

    def report(self, *args):
        for obj in args:
            fname = type(obj).__name__.lower()
            obj.create_file(fname, self._epochs, self._optimizer, self._loss_func, self._accuracy, )
