""" uncertainty_sim.py

Simulate neural network uncertainty.

Copyright (C) 2021 Steven Wong <steven.ykwong87@gmail.com>

MIT License

"""

import numpy as np
from numpy.random import default_rng
import pandas as pd
import pickle
import matplotlib.pyplot as plt


"""
Random noise
"""

rng = default_rng()
T = 100
N = 10
M = 5

X = rng.normal(0., 1., size=(N, N, M))
X = np.tile(X, (int(T / N), 1, 1))

covs = []
ys = []
errors = []
for t in range(T):
	cov = 0.1 * X[t,:,[0]].T @ X[t,:,[0]]
	# cov = np.tril(cov)
	# cov = cov @ cov.T
	d = np.log(1. + np.exp(X[t,:,:].mean(axis=-1) - 2))
	np.fill_diagonal(cov, d)
	covs.append(cov)

	e = rng.multivariate_normal(np.zeros(N), cov)
	y = np.tanh(X[t,:,:].sum(axis=1)) + e
	ys.append(y)
	errors.append(e)

X = X.astype('float32')
ys = np.stack(ys, axis=0).astype('float32')
ys = np.expand_dims(ys, axis=-1)
errors = np.stack(errors, axis=0).astype('float32')
covs = np.stack(covs, axis=0).astype('float32')


pickle.dump((X, ys, covs, errors), open('sim.pkl', 'wb'))
