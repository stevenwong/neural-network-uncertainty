""" ts_uncertainty_sim.py

Simulate neural network uncertainty.

Copyright (C) 2021 Steven Wong <steven.ykwong87@gmail.com>

MIT License

"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

np.random.seed(seed=42)

N = 1000
M = 1
T = 2000

X = np.tile(np.arange(T), (N, 1)) / T * 20 * np.pi + np.random.random((N, 1)) * 20 * np.pi
Y = np.sin(X / np.pi) + np.random.normal(0., 0.5, size=(N, T)) * np.sin(X / np.pi)


X = X.astype('float32')
Y = Y.astype('float32')
Y = np.expand_dims(Y, axis=-1)

plt.plot(X[0], Y[0], '.')
plt.show()
plt.cla()

pickle.dump(Y, open('uncertain_sim.pkl', 'wb'))
