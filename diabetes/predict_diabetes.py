from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import train_test_split
import numpy

numpy.random.seed(10)

# loading the dataset
data = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = data[:, 0:8] # Variables
Y = data[:, 8] # Target

# creating the model
model = Sequential()
model.add(Dense(12, input_dim=8, init='normal', activation='relu'))
model.add(Dense(8, init='normal', activation='relu'))
model.add(Dense(1, init='normal', activation='sigmoid'))

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.33, random_state=10)
# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.fit(X, Y, validation_split=0.33, nb_epoch=100, batch_size=10)
model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), nb_epoch=100, batch_size=10)

scores = model.evaluate(X, Y) # evaluate on train itself
print("Accuracy: %.2f%%" % (scores[1]*100))
