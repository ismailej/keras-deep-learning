from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy
import pandas

def baseline_model():
  model = Sequential()
  model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
  model.add(Dense(3, init='normal', activation='sigmoid'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  
  return model

seed = 7
numpy.random.seed(seed)

#Loading the dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

# The output is string class, convert it to one hot encoding using LabelEncoder

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Conver to one hot encoding
dummy_y = np_utils.to_categorical(encoded_Y)

model = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
kfold = KFold(n=len(dummy_y), n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
