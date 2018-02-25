import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

print('get data')

l_data_sets = ['r1_t1', 'r1_t1_d', 'r1_t2', 'r1_t2_d',
               'r1_t2_b', 'r1_t2_b2', 'r1_t2_e', 'r1_t2_e2', 'r1_t2_s', 'r1_t2_s2',
               'r1_t2_b_e_s', 'r1_t2_b_e_s2',
               'r2_t1', 'r2_t1_d', 'r2_t2', 'r2_t2_d',
               'r2_t2_b', 'r2_t2_b2', 'r2_t2_e', 'r2_t2_e2', 'r2_t2_s', 'r2_t2_s2',
               'r2_t2_b_e_s', 'r2_t2_b_e_s2',
               ]

for data_set in l_data_sets:

    print('data set: ' + data_set)

    train_df = pd.read_csv("../../input/paper2/dataset_" + data_set + "_train.csv")
    test_df = pd.read_csv("../../input/paper2/dataset_" + data_set + "_test.csv")

    col = [c for c in train_df.columns if c not in ['class_y0']]

    x_train = train_df[col]
    y_train = train_df['class_y0']
    x_test = test_df[col]
    y_test = test_df['class_y0']

    print(x_train.shape, '|', x_test.shape)

    print('pre-process data')

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print('train mlp')

    columns = ['solver', 'activation', 'h1_neurons', 'h2_neurons', 'ep', 'momentum', 'lr', 'auc_mean', 'auc_std',
               'f-measure_mean', 'f-measure_std', 'accuracy_mean', 'accuracy_std',
               'kappa_mean', 'kappa_std', 'mcc_mean', 'mcc_std']
    df = pd.DataFrame(columns=columns)

    print(columns)

    solver = 'lbfgs'
    activation = 'relu'
    l_h1_neurons = np.arange(10, 105, 10)
    l_ep = np.array([50, 100, 500, 1000])
    l_mc = np.arange(0.1, 1, 0.1)
    lr = 0.1
    count = 0

    for h1 in np.nditer(l_h1_neurons):
        for ep in np.nditer(l_ep):
            for mc in np.nditer(l_mc):
                columns_aux = ['solver', 'activation', 'h1_neurons', 'h2_neurons', 'ep', 'momentum', 'lr', 'auc',
                               'f-measure', 'accuracy', 'kappa', 'mcc']
                df_aux = pd.DataFrame(columns=columns_aux)
                count_aux = 0
                for i in np.arange(1, 6):
                    clf = MLPClassifier(solver=solver, activation=activation, hidden_layer_sizes=(h1,),
                                        learning_rate_init=lr, max_iter=ep, momentum=mc, random_state=i)
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)
                    fpr, tpr, thr = metrics.roc_curve(y_test, y_pred, pos_label=1)
                    auc = metrics.auc(fpr, tpr)
                    f1_score = metrics.f1_score(y_test, y_pred)
                    acc = metrics.accuracy_score(y_test, y_pred)
                    kappa = metrics.cohen_kappa_score(y_test, y_pred)
                    mcc = metrics.matthews_corrcoef(y_test, y_pred)
                    results_aux = [solver, activation, h1, 0, ep, mc, lr, auc, f1_score, acc, kappa, mcc]
                    df_aux.loc[count_aux] = results_aux
                    count_aux += 1

                results = [solver, activation, h1, 0, ep, mc, lr,
                           df_aux['auc'].mean(), df_aux['auc'].std(),
                           df_aux['f-measure'].mean(), df_aux['f-measure'].std(),
                           df_aux['accuracy'].mean(), df_aux['accuracy'].std(),
                           df_aux['kappa'].mean(), df_aux['kappa'].std(),
                           df_aux['mcc'].mean(), df_aux['mcc'].std()]
                df.loc[count] = results
                count += 1
                print(results)

    print('generate file results')

    df.to_csv('../../output/paper2/' + data_set + '_mlp_results.csv', encoding='utf-8')
