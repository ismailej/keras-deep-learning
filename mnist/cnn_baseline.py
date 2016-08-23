import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

seed = 7
np.random.seed(seed)

(X_train, X_label), (Y_test, Y_label) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]).astype(float)
Y_test = Y_test.reshape(Y_test.shape[0], 1, Y_test.shape[1], Y_test.shape[2]).astype(float)

#normalizing
X_train /= 255
Y_test /= 255

# one hot encoding
X_label = np_utils.to_categorical(X_label)
Y_label = np_utils.to_categorical(Y_label)
num_class = X_label.shape[1]
print(num_class)

def baseline_model():
    global num_class
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, init='normal', activation='relu'))
    model.add(Dense(num_class, init='normal', activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = baseline_model()

model.fit(X_train, X_label, validation_data=(Y_test, Y_label), nb_epoch=10, batch_size=200, verbose=2)
model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-result[1]*100))
