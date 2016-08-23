import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("sonar.csv", header=None)
data = dataframe.values

X = data[:, 0:60].astype(float)
Y = data[:, 60]
#print(Y[0])
encoder = LabelEncoder()
encoder.fit(Y)
encoded_y = encoder.transform(Y)
#print(encoded_y)

def baseline():
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(60, input_dim = 60, init='normal', activation='relu'))
    model.add(Dense(30, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))

    sgd = SGD(lr=0.1, momentum=0.8, decay=0.002, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=baseline, nb_epoch=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=encoded_y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

