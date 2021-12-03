""" ts_uncertainty_train.py

Simulate neural network uncertainty.

Copyright (C) 2021 Steven Wong <steven.ykwong87@gmail.com>

MIT License

"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

# TF imports
import tensorflow as tf
from tensorflow.python.keras.backend import square
from tensorflow.python.ops.gen_batch_ops import Batch
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
	try:
		tf.config.experimental.set_memory_growth(physical_devices[0], True)
	except:
		print('Could not set GPU memory growth')
		raise

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization
from tensorflow.keras.layers import Input, Lambda, Activation
from tensorflow.keras.optimizers import Adam

from network import *


# for manual training loop
from sklearn.metrics import mean_squared_error


X, y, covs, errors = pickle.load(open('sim.pkl', 'rb'))

T, N, input_dim = X.shape
# split data into 6 / 2 / 2
train_split = int(T * 0.6)
test_split = int(T * 0.8)


train_X = X[:train_split,:,:]
train_y = y[:train_split,:]
train_covs = covs[:train_split,:,:]
train_errors = errors[:train_split,:]

test_X = X[train_split:test_split,:,:]
test_y = y[train_split:test_split,:]
test_covs = covs[train_split:test_split,:,:]
test_errors = errors[train_split:test_split,:]

predict_X = X[test_split:,:,:]
predict_y = y[test_split:,:]
predict_covs = covs[test_split:,:,:]
predict_errors = errors[test_split:,:]

train_X = np.concatenate(train_X, axis=0)
train_y = np.concatenate(train_y, axis=0)
test_X = np.concatenate(test_X, axis=0)
test_y = np.concatenate(test_y, axis=0)


def forecast_gaussian(model, predict_X, predict_y):
	T, N, M = predict_X.shape
	y_true = []
	y_pred = []
	y_var = []
	for t in range(T):
		y_true.append(predict_y[t,:])
		X = predict_X[t,:,:]
		u, v = model.predict(X, batch_size=N)
		y_pred.append(u)
		y_var.append(v)

	return (np.stack(y_true, axis=0),
			np.stack(y_pred, axis=0),
			np.stack(y_var, axis=0))


"""
MVN DNN. In an ensemble, mean is the mean of ensemble. Var(X) = 1/M \sum_m [Var_m(X) + u_m(X)^2] - u(X)^2.
See `https://math.stackexchange.com/questions/195911/calculation-of-the-covariance-of-gaussian-mixtures`
"""
def build_mvn(input_dim,
			  output_dim,
			  cov_dim,
			  hidden_units=[8],
			  hidden_activation='relu',
			  learning_rate=0.01):
	nn = input_layer = Input(shape=input_dim)

	for u in hidden_units:
		nn = Dense(u, activation=hidden_activation)(nn)

	output_layer = MultivariateGaussian(cov_dim, output_dim)(nn)

	model = keras.Model(input_layer, output_layer)
	opt = Adam(learning_rate=learning_rate)
	model.compile(loss='mse', optimizer=opt)

	return model, opt


def build_train_mvn(params, train_X, train_y, test_X, test_y, metric):
	input_dim = train_X.shape[-1]
	output_dim = train_y.shape[-1]
	model, opt = build_mvn(input_dim, output_dim, **params)
	return train_model(model,
		opt,
		metric,
		metric,
		train_X,
		train_y,
		test_X,
		test_y,
		batch_size=10,
		max_epochs=100,
		hp_mode=True)


best_params, scores = hyperparameter_search({'hidden_units': [[8], [16, 8], [32, 16, 8]], 'cov_dim': [1, 2, 4, 8]},
	build_train_mvn, train_X, train_y, test_X, test_y, var_mse, n_cpu=1)

y_true = []
y_pred = []
cov_pred = []
for i in range(10):
	model, opt = build_mvn(input_dim, 1, **best_params)
	model = train_model(model,
		opt,
		var_mse,
		var_mse,
		train_X,
		train_y,
		test_X,
		test_y,
		batch_size=10,
		max_epochs=100)

	a, b, c = forecast_gaussian(model, predict_X, predict_y)
	y_true.append(a)
	y_pred.append(b)
	cov_pred.append(c)

y_true = y_true[0]
y_pred = np.stack(y_pred, axis=0)
cov_pred = np.stack(cov_pred, axis=0)

u = []
for i in range(y_true.shape[0]):
	for m in range(10):
		y_ = y_pred[m,i,:,:]
		v = y_ @ y_.T
		cov = cov_pred[m,i,:,:]
		cov_pred[m,i,:,:] = cov + v

y_pred = y_pred.mean(axis=0)
cov_pred = cov_pred.mean(axis=0)
cov_true = forecast_error(y_true, y_pred, var_type='y')

for i in range(y_true.shape[0]):
	y_ = y_pred[i,:,:]
	# Cov(y|X) = E(y @ y.T) - E(y|X) @ E(y|X).T
	# y_ = F(X,theta)
	cov_pred[i,:,:] = cov_pred[i,:,:] - y_ @ y_.T

scores = score(y_true, y_pred, cov_true, cov_pred)

empirical_cov = np.cov(y[:80,:,:].squeeze(), rowvar=False)
empirical_scores = score(y_true, y_pred, cov_true, np.tile(np.expand_dims(empirical_cov, axis=0), (20, 1, 1)))


"""
Gaussian DNN
"""
def build_gaussian(input_dim,
				   output_dim,
				   hidden_units=[8],
				   hidden_activation='relu'):
	nn = input_layer = Input(shape=input_dim)

	for u in hidden_units:
		nn = Dense(u, activation=hidden_activation)(nn)

	output_layer = GaussianLayer(output_dim)(nn)

	model = keras.Model(input_layer, output_layer)
	opt = Adam(learning_rate=0.01)
	model.compile(loss='mse', optimizer=opt)

	return model, opt

# predict for each period
model, opt = build_gaussian(input_dim, 1, [16, 8])

model = train_model(model,
	opt,
	nll,
	nll,
	np.concatenate(train_X, axis=0),
	np.concatenate(train_y, axis=0),
	np.concatenate(test_X, axis=0),
	np.concatenate(test_y, axis=0),
	batch_size=10,
	max_epochs=100)

y_true, y_pred, y_var = forecast_gaussian(model, predict_X, predict_y, multivariate=False)
var_true = np.concatenate([np.diag(v) for v in covs[-20:]], axis=0)


"""
Test vanilla DNN
"""
def build_dnn(input_dim,
			  output_dim,
			  hidden_units=[8],
			  hidden_activation='relu'):
	nn = input_layer = Input(shape=input_dim)

	for u in hidden_units:
		nn = Dense(u, activation=hidden_activation)(nn)

	output_layer = Dense(output_dim, activation='linear')(nn)

	model = keras.Model(input_layer, output_layer)
	opt = Adam(learning_rate=0.01)
	model.compile(loss='mse', optimizer=opt)

	return model, opt


model, opt = build_dnn(input_dim, 1, [8, 4])

model = train_model(model,
	opt,
	mse,
	mse,
	np.concatenate(train_X, axis=0),
	np.concatenate(train_y, axis=0),
	np.concatenate(test_X, axis=0),
	np.concatenate(test_y, axis=0),
	batch_size=10,
	max_epochs=100)


