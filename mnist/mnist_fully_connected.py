from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.layers import Dropout
from keras.utils import np_utils
import numpy as np

seed = 7
np.random.seed(seed)

# loading the dataset
(X_train, X_label), (Y_test, Y_label) = mnist.load_data()
print(X_train.shape, Y_test.shape, X_label.shape, type(X_train))
# flatten the vectors for passing through the first hidden layer

num_classes = len(np.unique(Y_label))
inputs = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
Y_test = Y_test.reshape((Y_test.shape[0], Y_test.shape[1]*Y_test.shape[2]))

X_train = X_train / 255
Y_train = Y_train / 255
X_label = np_utils.to_categorical(X_label)
Y_label = np_utils.to_categorical(Y_label)

def create_baseline():
    global inputs
    model = Sequential()
    model.add(Dense(inputs, input_dim=inputs, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='sigmoid'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_baseline()

model.fit(X_train, X_label,nb_epoch=2, batch_size=28, verbose=1)
result = model.evaluate(Y_test, Y_label)
print("Baseline Error: %.2f%%" % (100-result[1]*100))
