import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

# conver the dataset to np.ndarray format
def create_dataset(dataset, look_back=1):
    X_train = []
    X_predict = []
    for i in range(look_back, len(dataset)):
        k = np.array(dataset[i - look_back: i])
        k = k.flatten()
        y = np.array(dataset[i])
        y = y.flatten()
        print("value",  k)
        X_train.append(k)
        X_predict.append(y)

    return np.array(X_train), np.array(X_predict)

dataframe = pandas.read_csv('international-airline-passengers.csv',usecols=[1],engine='python',  skipfooter=3) 
data = dataframe.values
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data)

look_back = 1
X_train, X_predict = create_dataset(data, look_back)
print(X_train[:5, :], X_predict[:5, :], X_train.shape, X_predict.shape)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1]) 
# Split to test and train 
l = int(len(X_train) * 0.67)

x_train = X_train[:l, :]
x_predict = X_predict[:l, :]

y_train = X_train[l:, :]
y_predict = X_predict[l:, :]

def create_baseline():
    model = Sequential()
    model.add(LSTM(4, input_dim=look_back))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = create_baseline()
model.fit(x_train, x_predict, nb_epoch=2, batch_size=1, verbose=2)
result = model.evaluate(y_train, y_predict)
print(result)
