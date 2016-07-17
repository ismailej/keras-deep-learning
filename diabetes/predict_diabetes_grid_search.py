from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import numpy

def create_model(optimizer='rmsprop', init='uniform'):
  model = Sequential()
  model.add(Dense(12, input_dim=8, init=init, activation='relu'))
  model.add(Dense(8, init=init, activation='relu'))
  model.add(Dense(1, init=init, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

numpy.random.seed(10)

dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

model = KerasClassifier(build_fn=create_model, verbose=0)
optimizers = ['rmsprop', 'adam']
epochs = numpy.array([50, 100, 150])
batch = numpy.array([5, 10, 20])
init = ['uniform', 'normal']
param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batch, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)
print("Best: %f using %s" %(grid_result.best_score_, grid_result.best_params_))

