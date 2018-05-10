from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import cv2

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

cnn = Sequential()

cnn.add(Dense(512, activation='relu', input_shape=(784,)))

cnn.add(Dense(512, activation='relu'))
cnn.add(Dense(10, activation='softmax'))
cnn.compile(optimizer='adam', loss="categorical_crossentropy",
            metrics=['accuracy'])
cnn.summary()
history = cnn.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

plt.plot(history.history['acc'], 'g-')
plt.plot(history.history['val_acc'], 'b-')

plt.plot(history.history['loss'], 'r-')
plt.xlim(0, 4)

tr = cv2.imread('train1.png', 0)
tr = cv2.resize(tr, (28, 28))
tr = tr.astype('float32')
t = tr/255
t = t.flatten()
t = np.array([t])
print(cnn.predict(t).argmax())
