from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
import numpy 
import pandas

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv('sonar.csv', header=None)
dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

encoder = LabelEncoder()
encoder.fit(Y)
dummy_y = encoder.transform(Y)

def model():
 model = Sequential()
 model.add(Dense(60, input_dim=60, init='normal', activation='relu'))
 model.add(Dense(30, init='normal', activation='relu'))
 model.add(Dense(1, init='normal', activation='sigmoid'))

 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

 return model

# Use pipe to Standardize the input
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=model, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=dummy_y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% " % (results.mean() * 100))

