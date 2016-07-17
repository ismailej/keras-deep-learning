from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
import numpy

def create_model():
  model = Sequential()
  model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
  model.add(Dense(8, init='uniform', activation='relu'))
  model.add(Dense(1, init='uniform', activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

numpy.random.seed(10)

dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

model = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=10, verbose=0)
kfold = StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=10)
scores = cross_val_score(model, X, Y, cv=kfold)
print(scores.mean())

