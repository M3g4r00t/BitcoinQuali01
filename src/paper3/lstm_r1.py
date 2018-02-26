from math import sqrt

import matplotlib
import numpy
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from pandas import DataFrame
from pandas import read_csv
from sklearn import metrics
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
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #print(model.summary())
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

    columns = ['epochs', 'neurons', 'auc',
               'f-measure', 'accuracy', 'kappa', 'mcc']

    df_aux = DataFrame(columns=columns)

    count_aux = 0

    for r in range(repeats):
        # fit the model
        batch_size = 10
        lstm_model = fit_lstm(train_X, train_y, batch_size, epochs, 100)
        # forecast the entire training dataset to build up state for forecasting
        lstm_model.predict(train_X, batch_size=batch_size)
        # forecast test dataset
        predictions = lstm_model.predict(test_X, batch_size=batch_size)
        # report performance
        rmse = sqrt(metrics.mean_squared_error(test_y, predictions))

        error_scores.append(rmse)

        predictions = numpy.array(predictions)
        predictions = predictions.astype(int)

        fpr, tpr, thr = metrics.roc_curve(test_y, predictions, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        f1_score = metrics.f1_score(test_y, predictions)
        acc = metrics.accuracy_score(test_y, predictions)
        kappa = metrics.cohen_kappa_score(test_y, predictions)
        mcc = metrics.matthews_corrcoef(test_y, predictions)

        results_aux = [epochs, 20, auc, f1_score, acc, kappa, mcc]
        df_aux.loc[count_aux] = results_aux
        count_aux += 1

        print('%d) Test RMSE: %.3f' % (r + 1, rmse), ' ACC: %.3f' % acc, ' AUC: %.3f' % auc)

    results = [epochs, 20,
               df_aux['auc'].mean(), df_aux['auc'].std(),
               df_aux['f-measure'].mean(), df_aux['f-measure'].std(),
               df_aux['accuracy'].mean(), df_aux['accuracy'].std(),
               df_aux['kappa'].mean(), df_aux['kappa'].std(),
               df_aux['mcc'].mean(), df_aux['mcc'].std()]

    return error_scores, results


# load dataset
series = read_csv('../../input/paper3/dataset_10Min_tp.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# experiment
repeats = 10
results = DataFrame()
# vary training epochs
epochs = [10, 50, 100]

columns = ['epochs', 'neurons', 'auc_mean', 'auc_std',
               'f-measure_mean', 'f-measure_std', 'accuracy_mean', 'accuracy_std',
               'kappa_mean', 'kappa_std', 'mcc_mean', 'mcc_std']

df_summary_results = DataFrame(columns=columns)
count = 0
for e in epochs:
    print('Epochs: ', e)
    results[str(e)], summary_results = experiment(repeats, series[:208000], e)
    df_summary_results.loc[count] = summary_results
    count += 1
# summarize results
print(results.describe())
# save boxplot
results.boxplot()
pyplot.savefig('../../output/paper3/dataset_10Min_tp_boxplot_rmse_epochs.png')
results.to_csv('../../output/paper3/dataset_10Min_tp_results_rmse_epochs.csv')
df_summary_results.to_csv('../../output/paper3/dataset_10Min_tp_summary_results_epochs.csv')
