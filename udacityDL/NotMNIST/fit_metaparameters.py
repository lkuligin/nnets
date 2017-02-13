# coding: utf-8
"""
Fits the best LR model metaparameters
"""

import numpy as np
from six.moves import cPickle as pickle
from sklearn.linear_model import LogisticRegression
import warnings

DATA_PATH = 'notMNIST_clean.pickle'


def load_data(path):
    """loads the picke model """
    data = {}
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            train_dataset = data['train_dataset']
            train_labels = data['train_labels']
            valid_dataset = data['valid_dataset']
            valid_labels = data['valid_labels']
            test_dataset = data['test_dataset']
            test_labels = data['test_labels']
        return train_dataset, train_labels, test_dataset, test_labels, \
            valid_dataset, valid_labels
    except Exception as e:
        print('Unable to load data from', path, ':', e)


if __name__ == '__main__':
    sample_size = 20000
    train_dataset, train_labels, test_dataset, test_labels, \
        valid_dataset, valid_labels = load_data(DATA_PATH)

    idx = np.random.randint(0, train_dataset.shape[0], sample_size)
    X_train = train_dataset[idx, :].reshape(sample_size, 28 * 28)
    Y_train = train_labels[idx]
    X_test = test_dataset.reshape(test_dataset.shape[0], 28 * 28)
    Y_test = test_labels
    best_c = None
    best_solver = None
    best_score = None
#    best_model = None
    for c_ in (0.2, 2., 0.2):
        for solver_ in ('newton-cg', 'lbfgs', 'liblinear', 'sag'):
            lr_model = LogisticRegression(C=c_, solver=solver_, n_jobs=4)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                lr_model.fit(X_train, Y_train)
            score = lr_model.score(X_test, Y_test)
            type_ = "score"
            if score > best_score:
                best_score = score
#                best_model = lr_model
                best_c = c_
                best_solver = solver_
                type_ = "best score"
            print """{3}: {0:.2f},best_c: {1:.2f}, solver: {2}""". \
                format(
                    score,
                    c_,
                    solver_,
                    type_
                )
