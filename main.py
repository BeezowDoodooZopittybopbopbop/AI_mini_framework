from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from keras.utils import np_utils
import numpy as np

from optimizers import Adam, SGD
from activation_functions import ReLU
from loss_functions import MSE
from layers import Linear, Activation
from network import NeuralNetwork
from file_writers import Json, XML


# load data
iris = load_iris()
x, y = np.array(iris.data), np.array(iris.target)

# split on training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# preprocess train and test data
x_train = x_train.reshape(x_train.shape[0], 1, 4)
x_train = x_train.astype('float32')
y_train = np_utils.to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 4)
x_test = x_test.astype('float32')
y_test = np_utils.to_categorical(y_test)

# optimizer
sgd = SGD(learning_rate=0.1)
adam = Adam(learning_rate=0.1)

# loss and activation function
mse = MSE()
relu = ReLU()

# types of files for report
json_format = Json()
xml_format = XML()

# network
net = NeuralNetwork()
net.add(Linear(4, 20))
net.add(Activation(relu))
net.add(Linear(20, 10))
net.add(Activation(relu))
net.add(Linear(10, 3))
net.add(Activation(relu))

# train
net.fit(x_train, y_train, epochs=50, optimizer=adam, loss_func=mse, callback=True)

print()

net.predict(x_test, y_test)

net.report(json_format, xml_format)
