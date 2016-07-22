from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy 

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

filepath='weights.best.hdf5'
history = model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10,  verbose=0)
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
