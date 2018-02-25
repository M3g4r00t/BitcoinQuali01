import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler

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

    print('train svm')

    columns = ['kernel', 'degree', 'gamma', 'c', 'auc', 'f-measure', 'accuracy', 'kappa', 'mcc']
    df = pd.DataFrame(columns=columns)

    print(columns)

    l_degree = np.array([1, 2, 3, 4])
    l_gamma = np.append(1 / len(col), np.arange(0.1, 1.1, 0.1))
    l_c = np.array([0.5, 1, 5, 10])
    count = 0

    for degree in np.nditer(l_degree):
        for gamma in np.nditer(l_gamma):
            for c in np.nditer(l_c):
                clf = svm.SVC(degree=degree, gamma=gamma, C=c, kernel='poly')
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                fpr, tpr, thr = metrics.roc_curve(y_test, y_pred, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                f1_score = metrics.f1_score(y_test, y_pred)
                acc = metrics.accuracy_score(y_test, y_pred)
                kappa = metrics.cohen_kappa_score(y_test, y_pred)
                mcc = metrics.matthews_corrcoef(y_test, y_pred)
                results = ['polynomial', degree, gamma, c, auc, f1_score, acc, kappa, mcc]
                df.loc[count] = results
                count += 1
                print(results)

    l_gamma = np.append(1 / len(col), np.arange(0.1, 10.1, 0.1))
    l_c = np.array([0.5, 1, 5, 10, 100])

    for gamma in np.nditer(l_gamma):
        for c in np.nditer(l_c):
            clf = svm.SVC(gamma=gamma, C=c, kernel='rbf')
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            fpr, tpr, thr = metrics.roc_curve(y_test, y_pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            f1_score = metrics.f1_score(y_test, y_pred)
            acc = metrics.accuracy_score(y_test, y_pred)
            kappa = metrics.cohen_kappa_score(y_test, y_pred)
            mcc = metrics.matthews_corrcoef(y_test, y_pred)
            results = ['rbf', 'na', gamma, c, auc, f1_score, acc, kappa, mcc]
            df.loc[count] = results
            count += 1
            print(results)

    print('generate file results')

    df.to_csv('../../output/paper2/' + data_set + '_svm_results.csv', encoding='utf-8')
