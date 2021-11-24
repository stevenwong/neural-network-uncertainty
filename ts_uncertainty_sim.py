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
from tensorflow.python.keras.backend import dtype


"""
Random noise
"""
rng = default_rng()

N = 10
M = 1
T = 2000

X = np.tile(np.arange(T), (N, 1)) / T * 30 # + np.atleast_2d(np.arange(N) * 2 / np.pi).T
X[int(N/2):,:] = X[int(N/2):,:] + np.pi
e = rng.normal(0., 0.1, size=(N, T)) * (np.abs(np.sin(X)))
Y = np.sin(X) + e

pd.DataFrame(Y.T).plot()
plt.show()
plt.clf()


X = X.astype('float32')
Y = Y.astype('float32')
Y = np.expand_dims(Y, axis=-1)

pickle.dump(Y, open('uncertain_sim.pkl', 'wb'))
pickle.dump(e, open('uncertain_e.pkl', 'wb'))


"""
Single pattern
"""
rng = default_rng()

N = 10
M = 1
T = 2000

X = np.tile(np.arange(T), (N, 1)) / T * 30 # + np.atleast_2d(np.arange(N) * 2 / np.pi).T
# X[int(N/2):,:] = X[int(N/2):,:] + np.pi
e = rng.normal(0., 0.1, size=(N, T)) * (np.abs(np.sin(X)))
Y = np.sin(X) + e

pd.DataFrame(Y.T).plot()
plt.show()
plt.clf()


X = X.astype('float32')
Y = Y.astype('float32')
Y = np.expand_dims(Y, axis=-1)

pickle.dump(Y, open('uncertain_sim3.pkl', 'wb'))
pickle.dump(e, open('uncertain_e3.pkl', 'wb'))


"""
Correlated noise
"""

rng = default_rng()

N = 10
M = 1
T = 2000

X = np.tile(np.arange(T), (N, 1)) / T * 30
X[int(N/2):,:] = X[int(N/2):,:] + np.pi

c = np.ones((N, N)) * -0.01
even_idx = (np.arange(10, dtype=int) % 2 == 0)
c[even_idx, :] = 0.01
c[:, even_idx] = 0.01
d = rng.gamma(1., 0.1, size=N) + 1e-6
np.fill_diagonal(c, d)
u = np.zeros(N)

e = rng.multivariate_normal(u, c, size=T).T * (np.abs(np.sin(X)))
Y = np.sin(X) + e

pd.DataFrame(Y.T).plot()
plt.show()
plt.clf()


X = X.astype('float32')
Y = Y.astype('float32')
Y = np.expand_dims(Y, axis=-1)

pickle.dump(Y, open('uncertain_sim2.pkl', 'wb'))
pickle.dump(e, open('uncertain_e2.pkl', 'wb'))

