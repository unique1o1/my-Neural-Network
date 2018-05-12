#%%
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np

import sys
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#%%
if not sys.argv.__len__() >= 2:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    num_classes = 10
    epochs = 5

    X_train = X_train.reshape(60000, 28*28)

    X_test = X_test.reshape(10000, 28*28)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255.0
    X_test /= 255.0
    y_test = to_categorical(y_test, num_classes)
    y_train = to_categorical(y_train, num_classes)

    model = Sequential()

    model.add(Dense(512, activation='relu', input_shape=(784,)))

    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss="categorical_crossentropy",
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, epochs=epochs,
                        validation_data=(X_test, y_test))

    plt.plot(history.history['acc'], 'g-')
    plt.plot(history.history['val_acc'], 'b-')

    plt.plot(history.history['loss'], 'r-')
    plt.xlim(0, 4)

else:
    from PIL import Image
    model = load_model('classifier.h5')
#%%
    tr = Image.open('train1.png').convert('L')
    tr = tr.resize((28, 28))
    tr = np.array(tr.getdata())

    tr = tr.astype('float32')
    t = tr/255

    t = np.array([t])

    print(model.predict(t).argmax())

    plt.plot(np.arange(-10, 10), np.arange(-10, 10)**2)

#%%
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(5, 5), input_shape=(
    28, 28, 1), padding='same', activation='relu'))
cnn.add(MaxPooling2D())
cnn.add(Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu'))
cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dense(10, activation='softmax'))
cnn.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()
