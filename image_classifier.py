from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_classes = 10
epochs = 3

X_train = X_train.reshape(60000, 28*28)
X_test = X_test.reshape(10000, 28*28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255.0
X_test /= 255.0
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)
