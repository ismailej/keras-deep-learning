from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
import numpy 
import pandas

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv('housing.csv', delim_whitespace=True, header=None)
dataset = dataframe.values
X = dataset[:, 0:13]
Y = dataset[:, 13]

def model():
 model = Sequential()
 model.add(Dense(20, input_dim=13, init='normal', activation='relu'))
 model.add(Dense(1, init='normal'))

 model.compile(loss='mean_squared_error', optimizer='adam')
 return model

# Use pipe to Standardize the input
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp',KerasRegressor(build_fn=model, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n=len(Y), n_folds=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Accuracy: %.2f%% " % (results.mean()))

