import numpy as np


# calculate gradient
def calculate_gradients(input_data, weights, loss):
    # db==loss
    input_loss = np.dot(loss, weights.T)
    dw = np.dot(input_data.T, loss)

    return input_loss, dw, loss


# get the true value
def get_value(array):
    arr = np.where(array==np.amax(array))
    return int(arr[0]) + 1
