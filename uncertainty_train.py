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
def forecast_gaussian(model, predict_X, predict_y, multivariate=False):
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

	if multivariate:
		y_var = np.stack(y_var, axis=-1)
	else:
		y_var = np.concatenate(y_var, axis=1)

	return (np.concatenate(y_true, axis=1),
			np.concatenate(y_pred, axis=1),
			y_var)


model, opt = build_gaussian(input_dim, 1, [8, 4])

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

