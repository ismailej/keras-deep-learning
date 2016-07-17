from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import StratifiedKFold
import numpy

numpy.random.seed(10)

# loading the dataset
data = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = data[:, 0:8] # Variables
Y = data[:, 8] # Target

kfold = StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=10)
cvscores = []
for (train, test) in kfold:
  # creating the model
  model = Sequential()
  model.add(Dense(12, input_dim=8, init='normal', activation='relu'))
  model.add(Dense(8, init='normal', activation='relu'))
  model.add(Dense(1, init='normal', activation='sigmoid'))

  # Compiling the model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  model.fit(X[train], Y[train], nb_epoch=100, batch_size=10, verbose=0)

  scores = model.evaluate(X[test], Y[test], verbose=0) 
  cvscores.append(scores[1] * 100) 
print("Accuracy: %.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
