from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
import cv2
import sys

# import matplotlib.pyplot as plt
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# num_classes = 10
# epochs = 3

# X_train = X_train.reshape(60000, 28*28)
# X_test = X_test.reshape(10000, 28*28)

# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')

# X_train /= 255.0
# X_test /= 255.0
# y_test = to_categorical(y_test, 10)
# y_train = to_categorical(y_train, 10)

# model = Sequential()

# model.add(Dense(512, activation='relu', input_shape=(784,)))

# model.add(Dense(512, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.compile(optimizer='adam', loss="categorical_crossentropy",
#               metrics=['accuracy'])
# model.summary()
# history = model.fit(X_train, y_train, epochs=5,
#                     validation_data=(X_test, y_test))

# plt.plot(history.history['acc'], 'g-')
# plt.plot(history.history['val_acc'], 'b-')

# plt.plot(history.history['loss'], 'r-')
# plt.xlim(0, 4)
model = load_model('classifier.h5')

tr = cv2.imread(sys.argv[1], 0)
tr = cv2.resize(tr, (28, 28))
tr = tr.astype('float32')
t = tr/255
t = t.flatten()
t = np.array([t])
print(model.predict(t).argmax())
