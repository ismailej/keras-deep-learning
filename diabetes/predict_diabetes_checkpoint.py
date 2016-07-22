from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
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
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10, callbacks=callbacks_list, verbose=0)

scores = model.evaluate(X, Y, verbose=0)

print('Accuracy:  %.2f ' % (scores*100))
# Create a new model from the best saved weights
#new_model = Sequential()
#new_model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
#new_model.add(Dense(8, init='uniform', activation='relu'))
#new_model.add(Dense(1, init='uniform', activation='sigmoid'))

#new_model.compile(loss='binary_crossentropy', optimizer='adam')
#new_model.load_weights('weights.best.hdf5')


#scores = new_model.evaluate(X, Y, verbose=0)

#print('Accuracy:  %.2f ' % (scores*100))
