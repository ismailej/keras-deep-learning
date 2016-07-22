from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy 
import os

dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')

X = dataset[:, 0:8]
Y = dataset[:, 8]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")

#Evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
scores = loaded_model.evaluate(X, Y, verbose=0)

print('Accuracy:  %.2f ' % (scores*100))
