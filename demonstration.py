import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from good_neighbors import predict

# Simulate data set
slope = 1.5
intercept = 1
N = 25
noise = 1

data = pd.DataFrame(index=range(N), columns=['x', 'y'])

rv = stats.norm(0, noise)
x_obs = np.array(range(N)) + np.array(stats.norm.rvs(size=N))
y_obs = intercept + slope * np.array(range(N)) + np.array(stats.norm.rvs(size=N))
data.loc[:, 'x'] = x_obs
data.loc[:, 'y'] = y_obs
print("The data set:")
print(data)

# Extract training set - randomly select 75 % of the observations to train on
training_set = data.sample(frac=0.75)
test_set = data.drop(index=training_set.index)  # the rest of the data set becomes the test set
print("The training set:")
print(training_set)
print("The test set")
print(test_set)

# Plot
X_train = training_set.loc[:, 'x']
y_train = training_set.loc[:, 'y']
X_test = test_set.loc[:, 'x']
y_test = test_set.loc[:, 'y']

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, constrained_layout=True)

methods = ["nearest_neighbor",
           "weighted_nearest_neighbors",
           "sqr_weighted_nearest_neighbors",
           "exp_weighted_nearest_neighbors"]
axs = [ax1, ax2, ax3, ax4]

for i in range(4):
    ax = axs[i]
    method = methods[i]
    ax.set_title(method)
    ax.scatter(X_train, y_train, s=5, color='blue', label="training set")
    ax.scatter(X_test, y_test, color='red', label="test set")
    ax.scatter(X_test, [predict(training_set, x, method=method) for x in X_test],
               color='green', label="predictions")
    ax.legend()

x_values = np.linspace(0, N, 1000)
for ax in axs:
    ax.plot(x_values, [predict(training_set, x, method="ols") for x in x_values],
            color='black', label="ols regr line")
    ax.legend()
plt.show()
