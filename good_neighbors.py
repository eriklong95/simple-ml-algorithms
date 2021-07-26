import numpy as np
from numpy.linalg import inv
import pandas as pd


def nearest_neighbor(training_set, x):  # training_set must be pd.DataFrame with columns 'x' and 'y'.
    ''' Predict response for given predictor x by finding data point in training set
        with predictor nearest x.
    '''
    index_of_nearest_x = abs(x - pd.to_numeric(training_set.loc[:, 'x'])).idxmin()
    return training_set.loc[index_of_nearest_x, 'y']


def weighted_nearest_neighbors(training_set, x):
    ''' Predict response for given predictor x by calculating a weighted sum
        of the responses in the training set.
    '''

    a = 1 / np.abs(x - pd.to_numeric(training_set.loc[:, 'x']))
    numerator = np.dot(a, pd.to_numeric(training_set.loc[:, 'y']))
    denominator = a.sum()
    return numerator / denominator


def sqr_weighted_nearest_neighbors(training_set, x):
    ''' Predict response for given predictor x by calculating a weighted sum
        of the responses in the training set.
    '''

    a = 1 / np.abs(x - pd.to_numeric(training_set.loc[:, 'x'])) ** 2
    numerator = np.dot(a, pd.to_numeric(training_set.loc[:, 'y']))
    denominator = a.sum()
    return numerator / denominator


def exp_weighted_nearest_neighbors(training_set, x):
    ''' Predict response for given predictor x by calculating a weighted sum
        of the responses in the training set.
    '''

    a = np.exp(-np.abs(x - pd.to_numeric(training_set.loc[:, 'x'])))
    numerator = np.dot(a, pd.to_numeric(training_set.loc[:, 'y']))
    denominator = a.sum()
    return numerator / denominator


def ols(training_set, x):
    n = len(training_set)
    xs = np.array(training_set.loc[:, 'x'])
    ys = np.array(training_set.loc[:, 'y'])

    matrix = np.empty(shape=(n, 2))
    matrix[:, 0] = [1 for _ in range(n)]
    matrix[:, 1] = xs

    hat_beta = inv(matrix.T.dot(matrix)).dot(matrix.T).dot(ys)

    return hat_beta.T.dot(np.array([1, x]))


def predict(training_set, x, method="nearest_neighbor"):
    if method == "nearest_neighbor":
        return nearest_neighbor(training_set, x)
    elif method == "weighted_nearest_neighbors":
        return weighted_nearest_neighbors(training_set, x)
    elif method == "sqr_weighted_nearest_neighbors":
        return sqr_weighted_nearest_neighbors(training_set, x)
    elif method == "exp_weighted_nearest_neighbors":
        return exp_weighted_nearest_neighbors(training_set, x)
    else:
        return ols(training_set, x)



