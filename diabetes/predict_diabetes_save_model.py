from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy 
import os

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')

X = dataset[:, 0:8]
Y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=0)

scores = model.evaluate(X, Y, verbose=0)

print('Accuracy:  %.2f ' % (scores*100))

model_json = model.to_json()
with open("model.json", 'w') as json_file:
  json_file.write(model_json)

model.save_weights("model.h5")
print("Saved the model")

