""" ts_uncertainty_sim.py

Simulate neural network uncertainty.

Copyright (C) 2021 Steven Wong <steven.ykwong87@gmail.com>

MIT License

"""

import numpy as np
from numpy.random import default_rng
import pandas as pd
import pickle
import matplotlib.pyplot as plt


rng = default_rng()

N = 10
M = 1
T = 2000

X = np.tile(np.arange(T), (N, 1)) / T * 30 # + np.atleast_2d(np.arange(N) * 2 / np.pi).T
Y = np.concatenate([
    np.sin(X[:int(N/2)]) + rng.normal(0., 0.2, size=(int(N/2), T)) * (np.abs(np.sin(X[:int(N/2)]))),
    np.cos(X[int(N/2):]) + rng.normal(0., 0.2, size=(int(N/2), T)) * (np.abs(np.cos(X[int(N/2):])))
], axis=0)

# c = rng.normal(0., 0.2, size=(N, N))
# c = c @ c.T
# d = rng.gamma(1., 0.1, size=N)
# c = c + np.diag(d)
# u = np.ones(N) * 0.01

# Y = rng.multivariate_normal(u, c, size=T).T * np.abs(np.sin(X / np.pi))
# Y = np.cumsum(Y, axis=1)

pd.DataFrame(Y.T).plot()
plt.show()
plt.clf()


X = X.astype('float32')
Y = Y.astype('float32')
Y = np.expand_dims(Y, axis=-1)

pickle.dump(Y, open('uncertain_sim.pkl', 'wb'))

