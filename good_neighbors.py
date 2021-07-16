import numpy as np
from numpy.linalg import inv
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def nearest_neighbors(training_set, x):  # training_set must be pd.DataFrame with columns 'x' and 'y'.
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


def exp_sqr_weighted_nearest_neighbors(training_set, x):
    ''' Predict response for given predictor x by calculating a weighted sum
        of the responses in the training set.
    '''

    a = np.exp(- np.abs(x - pd.to_numeric(training_set.loc[:, 'x'])) ** 2)
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

if __name__ == '__main__':

    # Simulate a training set.
    slope = 1.5
    intercept = 1
    N = 25
    training_set = pd.DataFrame(index=range(N), columns=['x', 'y'])

    rv = stats.norm(0, 10)
    x_obs = np.array(range(N)) + np.array(stats.norm.rvs(size=N))
    y_obs = intercept + slope * np.array(range(N)) + np.array(stats.norm.rvs(size=N))
    training_set.loc[:, 'x'] = x_obs
    training_set.loc[:, 'y'] = y_obs
    print(training_set)

    print(ols(training_set, 10))

    # Plot data set and predictions.
    fig, ax = plt.subplots()

    # Training set
    ax.scatter(x_obs, y_obs)

    # Predictions
    x_values = np.linspace(0, N, 100)
    # ax.scatter(x_values, [nearest_neighbors(training_set, x) for x in x_values], color='green', s=1)
    # ax.scatter(x_values, [weighted_nearest_neighbors(training_set, x) for x in x_values], color='red', s=1)
    # ax.scatter(x_values, [sqr_weighted_nearest_neighbors(training_set, x) for x in x_values], color='purple', s=1)
    ax.scatter(x_values, [exp_weighted_nearest_neighbors(training_set, x) for x in x_values], color='yellow', s=1)
    ax.scatter(x_values, [exp_sqr_weighted_nearest_neighbors(training_set, x) for x in x_values], color='pink', s=1)
    ax.plot(x_values, [ols(training_set, x) for x in x_values], color='black')
    plt.show()


