import numpy 
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb

seed =  7
numpy.random.seed(seed)

top_words = 5000
(X_train, X_label), (Y_train, Y_label) = imdb.load_data(nb_words=top_words)
#print(len(X_train), len(X_label), len(Y_train))
max_words = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_words)
Y_train = sequence.pad_sequences(Y_train, maxlen=max_words)

model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(LSTM(4))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, X_label, nb_epoch=1, batch_size=1, verbose=2)

