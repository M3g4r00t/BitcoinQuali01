from math import sqrt

import matplotlib
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from pandas import DataFrame
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# be able to save images on server
matplotlib.use('Agg')
from matplotlib import pyplot


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# fit an LSTM network to training data
def fit_lstm(train_X, train_Y, batch_size, nb_epoch, neurons):
    X, y = train_X, train_Y
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# run a repeated experiment
def experiment(repeats, series, epochs):
    # assume dataset has time-series supervised form
    supervised_values = series.values
    # split data into train and test-sets
    n_train_hours = 156000
    train = supervised_values[:n_train_hours, :]
    test = supervised_values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # transform the scale of the data
    scaler, train_X, test_X = scale(train_X, test_X)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # run experiment
    error_scores = list()
    for r in range(repeats):
        # fit the model
        batch_size = 4
        lstm_model = fit_lstm(train_X, train_y, batch_size, epochs, 1)
        # forecast the entire training dataset to build up state for forecasting
        lstm_model.predict(train_X, batch_size=batch_size)
        # forecast test dataset
        output = lstm_model.predict(test_X, batch_size=batch_size)
        predictions = list()
        # report performance
        rmse = sqrt(mean_squared_error(test_y, predictions))
        print('%d) Test RMSE: %.3f' % (r + 1, rmse))
        error_scores.append(rmse)
    return error_scores


# load dataset
series = read_csv('../../input/paper3/dataset_10Min_tp.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# experiment
repeats = 30
results = DataFrame()
# vary training epochs
epochs = [500, 1000, 2000, 4000, 6000]
for e in epochs:
    results[str(e)] = experiment(repeats, series, e)
# summarize results
print(results.describe())
# save boxplot
results.boxplot()
pyplot.savefig('../../output/paper3/dataset_10Min_tp_boxplot_epochs.png')
