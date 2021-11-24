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
import math


def create_sequences(data,
					 seq_len,
					 forward,
					 stride,
					 debug=False):
	""" Create training and test sequences.

	Args:
		data (numpy.array): Assumed to be of shape (N, T, M).
		seq_len (int): Sequence length.
		forward (int): Predict forward N periods.
		stride (int): Shift by k amounts.

	"""
	X = []
	y = []

	N, T, M = data.shape
	for i in range(seq_len, T - forward + 1, stride):
		X.append(data[:, i - seq_len:i, :])
		# -1 because python slicing excludes end point
		y.append(data[:, i + forward - 1, :])

		if debug:
			print(f'X from {i - seq_len} to {i - 1}, y at {i + forward - 1}')

	return np.concatenate(X), np.concatenate(y)


X = pickle.load(open('uncertain_sim3.pkl', 'rb'))
e = pickle.load(open('uncertain_e3.pkl', 'rb'))

e2 = []
for i in range(e.shape[-1]):
	v = e[:,[i]]
	e2.append(v @ v.T)
e2 = np.stack(e2, axis=-1)

# constants
N, T, input_dim = X.shape
seq_len = 50
forward = 10
stride = 10


# split history into 50% train, 20% validate, 30% test
train_split = int(T * 0.5)
test_split = int(T * (0.5 + 0.2))
train_data = X[:, :train_split, :]
test_data = X[:, train_split:test_split, :]
predict_data = X[:, test_split:, :]

train_X, train_y = create_sequences(train_data, seq_len, forward, stride)
test_X, test_y = create_sequences(test_data, seq_len, forward, stride)
predict_X, predict_y = create_sequences(predict_data, seq_len, forward, stride)


# build model
def build_tcn(seq_len,
			  input_dim,
			  output_dim,
			  dilation_rates,
			  filters=8,
			  kernel_size=5,
			  hidden_units=[8]):
	nn = input_layer = Input(shape=(seq_len, input_dim))

	for i, d in enumerate(dilation_rates):
		nn = Conv1D(dilation_rate=d,
					filters=filters,
					kernel_size=kernel_size,
					padding='causal')(nn)
		nn = BatchNormalization()(nn)
		nn = Activation('relu')(nn)

	nn = latent = Lambda(lambda x: x[:, -1, :])(nn)

	for u in hidden_units:
		nn = Dense(u, activation='relu')(nn)

	output_layer = Dense(output_dim, activation='linear')(nn)

	model = keras.Model(input_layer, output_layer)
	opt = Adam(lr=0.01)
	model.compile(loss='mse', optimizer=opt)

	return model, opt


def build_gaussian_tcn(seq_len,
					   input_dim,
					   output_dim,
					   dilation_rates,
					   filters=8,
					   kernel_size=5,
					   hidden_units=[8]):
	nn = input_layer = Input(shape=(seq_len, input_dim))

	for i, d in enumerate(dilation_rates):
		nn = Conv1D(dilation_rate=d,
					filters=filters,
					kernel_size=kernel_size,
					padding='causal')(nn)
		nn = BatchNormalization()(nn)
		nn = Activation('relu')(nn)

	nn = latent = Lambda(lambda x: x[:, -1, :])(nn)

	for u in hidden_units:
		nn = Dense(u, activation='relu')(nn)

	output_layer = GaussianLayer(output_dim)(nn)

	model = keras.Model(input_layer, output_layer)
	opt = Adam(lr=0.01)
	model.compile(loss=nll, optimizer=opt)

	return model, opt


# negative log-likelihood
# https://stats.stackexchange.com/questions/351549/maximum-likelihood-estimators-multivariate-gaussian
# https://stats.stackexchange.com/questions/321152/covarince-matrix-has-to-be-be-rank-deficient-to-maximize-multivariate-gaussian-l
# https://math.stackexchange.com/questions/3124194/determinant-of-covariance-matrix
# https://math.stackexchange.com/questions/1479483/when-does-the-inverse-of-a-covariance-matrix-exist
# For the covariance matrix to be invertible, it has to be positive definite (not PSD). Meaning, there exists
# a \in R^n, such that Cov(s)a = 0. There exists some linear combination of X which has zero variance.
def nll_mvn(y, out, tape=None):
	u, v = out
	# NLL = log(var(x))/2 + (y - u(x))^2 / 2var(x)
	e = y - u
	return tf.reduce_mean(0.5 * tf.math.log(tf.linalg.det(v)) +
		0.5 * K.dot(K.dot(K.transpose(e), tf.linalg.inv(v)), e))


def squared_error(y, out, tape=None):
	# see `https://github.com/sthorn/deep-learning-explorations/blob/master/predicting-uncertainty-variance.ipynb`
	u, v = out
	e = y - u
	# print('v', tf.linalg.det(v))
	if tape is not None:
		with tape.stop_recording():
			V = K.dot(e, K.transpose(e))
			# V = tf.square(e)
	else:
		V = K.dot(e, K.transpose(e))
		# V = tf.square(e)
	return tf.reduce_mean(tf.square(e)) + 10. * tf.reduce_mean(tf.square(V - v))

record = []
def cov_squared_error(y, v, tape=None):
	# print('v', tf.linalg.det(v))
	if tape is not None:
		with tape.stop_recording():
			V = K.dot(y, K.transpose(y))
			record.append(V.numpy())
	else:
		V = K.dot(y, K.transpose(y))
	return tf.reduce_mean(tf.square(V - v))


def univariate_mse(y, out, tape=None):
	u, v = out
	e = y - u
	if tape is not None:
		with tape.stop_recording():
			V = K.square(e)
	else:
		V = K.square(e)
	return tf.reduce_mean(tf.square(e)) + tf.reduce_mean(tf.square(V - v))


def build_mvn_tcn(seq_len,
				  input_dim,
				  output_dim,
				  dilation_rates,
				  filters=8,
				  kernel_size=5,
				  hidden_units=[8]):
	nn = input_layer = Input(shape=(seq_len, input_dim))

	for i, d in enumerate(dilation_rates):
		nn = Conv1D(dilation_rate=d,
					filters=filters,
					kernel_size=kernel_size,
					padding='causal')(nn)
		nn = BatchNormalization()(nn)
		nn = Activation('relu')(nn)

	nn = Lambda(lambda x: x[:, -1, :])(nn)

	for u in hidden_units:
		nn = Dense(u, activation='relu')(nn)

	output_layer = MultivariateGaussian(output_dim)(nn)

	model = keras.Model(input_layer, output_layer)
	opt = Adam(lr=0.01)
	model.compile(loss=nll_mvn, optimizer=opt)

	return model, opt


def build_cov_tcn(seq_len,
				  input_dim,
				  output_dim,
				  dilation_rates,
				  filters=8,
				  kernel_size=5,
				  hidden_units=[8]):
	nn = input_layer = Input(shape=(seq_len, input_dim))

	for i, d in enumerate(dilation_rates):
		nn = Conv1D(dilation_rate=d,
					filters=filters,
					kernel_size=kernel_size,
					padding='causal')(nn)
		nn = BatchNormalization()(nn)
		nn = Activation('relu')(nn)

	nn = Lambda(lambda x: x[:, -1, :])(nn)

	for u in hidden_units:
		nn = Dense(u, activation='relu')(nn)

	output_layer = CovarianceMatrix(output_dim)(nn)

	model = keras.Model(input_layer, output_layer)
	opt = Adam(lr=0.01)
	model.compile(loss=nll_mvn, optimizer=opt)

	return model, opt


# predict for each period
def forecast(model, predict_data, seq_len=50, forward=10, stride=1):
	"""Step through the out-of-sample data and make predictions.

	Args:
		model (keras.Model): Trained model.
		predict_data (numpy.array): Out-of-sample data.
		seq_len (int): Sequence length.
		forward (int): Forward.
		stride (int): Step size.

	Returns:
		numpy.array: Predicted values.

	"""

	T = predict_data.shape[1]
	y_true = []
	y_pred = []
	for t in range(seq_len, T - forward, stride):
		y_true.append(predict_data[:,t + forward - 1,:])
		y_ = model.predict(predict_data[:,t-seq_len:t,:])
		y_pred.append(y_)

	return np.concatenate(y_true, axis=1), np.concatenate(y_pred, axis=1)


# predict for each period
def forecast_gaussian(model, predict_data, seq_len=50, forward=10, stride=1, multivariate=False):
	"""Step through the out-of-sample data and make predictions.

	Args:
		model (keras.Model): Trained model.
		predict_data (numpy.array): Out-of-sample data.
		seq_len (int): Sequence length.
		forward (int): Forward.
		stride (int): Step size.

	Returns:
		numpy.array: Predicted values.

	"""

	T = predict_data.shape[1]
	y_true = []
	y_pred = []
	y_var = []
	for t in range(seq_len, T - forward, stride):
		y_true.append(predict_data[:,t + forward - 1,:])
		X = predict_data[:,t-seq_len:t,:]
		u, v = model.predict(X, batch_size=X.shape[0])
		y_pred.append(u)
		y_var.append(v)

	if multivariate:
		y_var = np.stack(y_var, axis=-1)
	else:
		y_var = np.concatenate(y_var, axis=1)

	return (np.concatenate(y_true, axis=1),
			np.concatenate(y_pred, axis=1),
			y_var)


def forecast_covariance(model, predict_data, seq_len=50, forward=10, stride=1, multivariate=False):
	"""Step through the out-of-sample data and make predictions.

	Args:
		model (keras.Model): Trained model.
		predict_data (numpy.array): Out-of-sample data.
		seq_len (int): Sequence length.
		forward (int): Forward.
		stride (int): Step size.

	Returns:
		numpy.array: Predicted values.

	"""

	T = predict_data.shape[1]
	y_true = []
	y_var = []
	for t in range(seq_len, T - forward, stride):
		y_ = predict_data[:,t + forward - 1,:]
		y_ = y_ @ y_.T
		y_true.append(y_)
		X = predict_data[:,t-seq_len:t,:]
		v = model.predict(X, batch_size=X.shape[0])
		y_var.append(v)

	y_true = np.stack(y_true, axis=-1)
	y_var = np.stack(y_var, axis=-1)
	return y_true, y_var


class EmpiricalCovariance():
	def __init__(self, forward):
		self.forward = forward

	def predict(self, X, batch_size=None):
		X = X[:,::self.forward,:]
		return np.cov(X.squeeze())


# build multivariate normal network
model, opt = build_mvn_tcn(seq_len,
	input_dim,
	input_dim,
	[1, 2, 4, 8],
	filters=16,
	kernel_size=5,
	hidden_units=[16, 8])

model = train_model(model,
	opt,
	squared_error,
	squared_error,
	train_X,
	train_y,
	test_X,
	test_y,
	batch_size=10,
	max_epochs=100)

y_true, y_pred, y_var = forecast_gaussian(model, predict_data, seq_len=seq_len,
	forward=forward, stride=1, multivariate=True)

x_axis = np.arange(y_true.shape[1])
fig, ax = plt.subplots()
i = 0
ax.plot(x_axis, y_true[i,:], label='true')
ax.plot(x_axis, y_pred[i,:], label='pred')
ax.plot(x_axis, y_pred[i,:] + np.sqrt(y_var[i,i,:]), label='pred+std', linestyle='dotted', color='red')
ax.plot(x_axis, y_pred[i,:] - np.sqrt(y_var[i,i,:]), label='pred-std', linestyle='dotted', color='red')
plt.tight_layout()
plt.show()
plt.clf()


# build covariance network
model, opt = build_cov_tcn(seq_len,
	input_dim,
	input_dim,
	[1, 2, 4, 8],
	filters=16,
	kernel_size=5,
	hidden_units=[32, 16])

model = train_model(model,
	opt,
	cov_squared_error,
	cov_squared_error,
	train_X,
	train_y,
	test_X,
	test_y,
	batch_size=10,
	max_epochs=100)

y_true, y_var = forecast_covariance(model, predict_data, seq_len=seq_len,
	forward=forward, stride=1, multivariate=True)

cnn_mse = np.mean((y_true - y_var) ** 2)

y_true_avg = np.stack([y_true[:,:,i:i+10].mean(axis=-1) for i in range(0, y_true.shape[-1], 10)], axis=-1)
y_var_avg = np.stack([y_var[:,:,i:i+10].mean(axis=-1) for i in range(0, y_var.shape[-1], 10)], axis=-1)

# approximate covariance empirically
em = EmpiricalCovariance(10)
_, empirical = forecast_covariance(em, predict_data, seq_len=seq_len,
	forward=forward, stride=1, multivariate=True)

em_mse = np.mean((y_true - empirical) ** 2)


x_axis = np.arange(y_true.shape[1])
fig, ax = plt.subplots()
i = 0
ax.plot(x_axis, y_true[i,:], label='true')
ax.plot(x_axis, y_pred[i,:], label='pred')
ax.plot(x_axis, y_pred[i,:] + np.sqrt(y_var[i,i,:]), label='pred+std', linestyle='dotted', color='red')
ax.plot(x_axis, y_pred[i,:] - np.sqrt(y_var[i,i,:]), label='pred-std', linestyle='dotted', color='red')
plt.tight_layout()
plt.show()
plt.clf()


# dummy example
M = 1
x = np.random.normal(0., 1., size=(100, M)).astype('float32')
y = np.atleast_2d(np.sin(x).sum(axis=-1) + np.random.normal(0., 0.1, size=100).astype('float32')).T

input_layer = Input(shape=(M))
output_layer = MultivariateGaussian(1)(input_layer)

model = keras.Model(input_layer, output_layer)
opt = Adam(lr=0.01)
model.compile(loss='mse', optimizer=opt)

model = train_model(model,
	opt,
	squared_error,
	squared_error,
	x[:50,:],
	y[:50,:],
	x[50:,:],
	y[50:,:],
	batch_size=10)


"""
Univariate Gaussian network example.
"""

model, opt = build_gaussian_tcn(seq_len,
	input_dim,
	input_dim,
	[1, 2, 4, 8],
	filters=16,
	kernel_size=5,
	hidden_units=[16, 8])

model = train_model(model,
	opt,
	nll,
	nll,
	train_X,
	train_y,
	test_X,
	test_y)


y_true, y_pred, y_var = forecast_gaussian(model, predict_data, seq_len=seq_len,
	forward=forward, stride=1)

x_axis = np.arange(y_true.shape[1])
fig, ax = plt.subplots()
ax.plot(x_axis, y_true[0,:], label='true')
ax.plot(x_axis, y_pred[0,:], label='pred')
ax.plot(x_axis, y_pred[0,:] + np.sqrt(y_var[0,:]), label='pred+std', linestyle='dotted', color='red')
ax.plot(x_axis, y_pred[0,:] - np.sqrt(y_var[0,:]), label='pred-std', linestyle='dotted', color='red')
plt.tight_layout()
plt.show()
plt.clf()



"""
Normal non-probabilistic network example.
"""
# receptive field = 1 + layers_per_block * (kernel - 1) * stacks * sum(dilation)
model, opt = build_tcn(seq_len,
	input_dim,
	input_dim,
	[1, 2, 4, 8],
	filters=8,
	kernel_size=5,
	hidden_units=[8])

model = train_model(model,
	opt,
	mse,
	mse,
	train_X,
	train_y,
	test_X,
	test_y)


y_true, y_pred = forecast(model, predict_data, seq_len=seq_len, forward=forward, stride=1)

x_axis = np.arange(y_true.shape[1])
fig, ax = plt.subplots()
ax.plot(x_axis, y_true[0,:], label='true')
ax.plot(x_axis, y_pred[0,:], label='pred')
plt.tight_layout()
plt.show()
plt.clf()

