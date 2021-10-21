""" ts_uncertainty_train.py

Simulate neural network uncertainty.

Copyright (C) 2020 Steven Wong <steven.ykwong87@gmail.com>

MIT License

"""


import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# TF imports
import tensorflow as tf
from tensorflow.python.keras.backend import dtype
from tensorflow.python.ops.gen_math_ops import Mul, mean
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

from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Layer
from tensorflow.keras.layers import Input, Lambda, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotNormal
import tensorflow_probability as tfp

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


X = pickle.load(open('uncertain_sim.pkl', 'rb'))

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


class GaussianLayer(Layer):
	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(GaussianLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		input_dim = input_shape[-1]
		self.kernel_1 = self.add_weight(name='kernel_1', 
									  shape=(input_dim, self.output_dim),
									  initializer=GlorotNormal(),
									  trainable=True)
		self.kernel_2 = self.add_weight(name='kernel_2', 
									  shape=(input_dim, self.output_dim),
									  initializer=GlorotNormal(),
									  trainable=True)
		self.bias_1 = self.add_weight(name='bias_1',
									shape=(self.output_dim, ),
									initializer=GlorotNormal(),
									trainable=True)
		self.bias_2 = self.add_weight(name='bias_2',
									shape=(self.output_dim, ),
									initializer=GlorotNormal(),
									trainable=True)
		super(GaussianLayer, self).build(input_shape) 

	def call(self, x):
		output_mu  = K.dot(x, self.kernel_1) + self.bias_1
		output_sig = K.dot(x, self.kernel_2) + self.bias_2
		output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06  
		return [output_mu, output_sig_pos]

	def compute_output_shape(self, input_shape):
		return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]


class MultivariateGaussian(Layer):
	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(MultivariateGaussian, self).__init__(**kwargs)

	def build(self, input_shape):
		input_dim = input_shape[-1]
		print(input_dim)
		# mu weights
		self.mu_kernel = self.add_weight(name='mu_kernel', 
										 shape=(input_dim, self.output_dim),
										 initializer=GlorotNormal(),
										 trainable=True)
		# covariance weights
		self.cov_kernel = self.add_weight(name='cov_kernel',
										  shape=(input_dim, input_dim),
										  initializer=GlorotNormal(),
										  trainable=True)
		# idiosyncratic variance kernel
		self.idio_kernel = self.add_weight(name='idio_kernel', 
										   shape=(input_dim, self.output_dim),
										   initializer=GlorotNormal(),
										   trainable=True)
		# mu bias
		self.mu_bias = self.add_weight(name='mu_bias',
									   shape=(self.output_dim, ),
									   initializer=GlorotNormal(),
									   trainable=True)
		# idiosyncratic variance bias
		self.idio_bias = self.add_weight(name='idio_bias',
										 shape=(self.output_dim, ),
										 initializer=GlorotNormal(),
										 trainable=True)
		# number of entries in lower triangular matrix is N(N + 1) / 2
		ones = tf.ones(int(input_dim * (input_dim + 1) / 2))
		self.mask = tfp.math.fill_triangular(ones)
		super(MultivariateGaussian, self).build(input_shape) 

	def call(self, x):
		cov = self.cov_kernel * self.mask
		cov = K.dot(cov, K.transpose(cov)) # + tf.linalg.diag(self.cov_bias)
		# cov = K.dot(x, self.cov_kernel) + self.cov_bias
		output_mu  = K.dot(x, self.mu_kernel) + self.mu_bias
		idio_var = K.dot(x, self.idio_kernel) + self.idio_bias
		idio_var = K.log(1 + K.exp(idio_var)) + 1e-08
		idio_var = tf.squeeze(idio_var)
		output_cov = K.dot(K.dot(x, cov), K.transpose(x)) + tf.linalg.diag(idio_var)
		# output_cov = K.log(1 + K.exp(cov)) + 1e-06
		return [output_mu, output_cov]

	def compute_output_shape(self, input_shape):
		return [(input_shape[0], self.output_dim), (input_shape[0], input_shape[0])]


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
	model.compile(loss='mse', optimizer=opt)

	return model, opt


# negative log-likelihood multivariate normal
def nll(y, out, tape=None):
	u, v = out
	# NLL = log(var(x))/2 + (y - u(x))^2 / 2var(x)
	return tf.reduce_mean(0.5 * tf.math.log(v) +
		0.5 * tf.math.divide(tf.math.square(y - u), v)) + 1e-6


# negative log-likelihood
def nll_mvn(y, out, tape=None):
	u, v = out
	# NLL = log(var(x))/2 + (y - u(x))^2 / 2var(x)
	e = y - u
	return tf.reduce_mean(0.5 * tf.math.log(tf.linalg.det(v)) +
		0.5 * K.dot(K.dot(K.transpose(e), tf.linalg.inv(v)), e)) + 1e-6


def squared_error(y, out, tape=None):
	u, v = out
	e = y - u
	if tape is not None:
		with tape.stop_recording():
			V = K.dot(e, K.transpose(e))
	else:
		V = K.dot(e, K.transpose(e))
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
	opt = Adam(lr=0.0001)
	model.compile(loss=nll_mvn, optimizer=opt)

	return model, opt


# def train_mse_cov(model, x, y):
# 	y_, exposures, cov = model(x)
# 	e = y - y_
# 	with tf.stop_recording():
# 		S = e @ e.T
# 		L = tf.linalg.cholesky(S)
# 		mask = np.tril(np.ones_like(L))

# 	return K.mean(K.square(e), axis=0) + K.mean(K.square(exposures @ L - exposures @ (mask * cov)))


def calc_gradient(model, x, y, loss_fn):
	with tf.GradientTape() as tape:
		y_ = model(x)
		loss = loss_fn(y, y_, tape=tape)
		grads = tape.gradient(loss, model.trainable_variables)
	return loss, grads


def train_model(model,
				optimiser,
				train_criterion,
				test_criterion,
				train_X,
				train_y,
				test_X,
				test_y,
				batch_size=1000,
				max_epochs=100,
				min_delta=0.0001,
				patience=10):
	N = train_X.shape[0]
	B = math.ceil(N / batch_size)

	# for early stopping
	best_loss = np.inf
	best_weights = None
	best_epochs = 0
	counter = 0

	test_X = tf.convert_to_tensor(test_X)
	test_y = tf.convert_to_tensor(test_y)

	for e in range(max_epochs):
		permutations = np.random.permutation(N)
		sum_loss = 0.

		for b in range(0, N, batch_size):
			idx = permutations[b:b + batch_size]
			batch_X = train_X[idx]
			batch_y = train_y[idx]

			loss, grads = calc_gradient(model, batch_X, batch_y, train_criterion)
			optimiser.apply_gradients(zip(grads, model.trainable_variables))
			loss = loss.numpy().item()
			sum_loss += loss

		sum_loss = sum_loss / B
		y_ = model.predict(test_X, batch_size=test_X.shape[0])
		val_loss = test_criterion(test_y, y_)
		if isinstance(val_loss, tf.Tensor):
			val_loss = val_loss.numpy().item()

		print(f'| Epoch {e}/{max_epochs} - loss: {loss:.4f} - val {val_loss:.4f}')

		if val_loss + min_delta >= best_loss:
			counter += 1
		else:
			counter = 0

		if val_loss < best_loss:
			best_loss = val_loss
			best_weights = model.get_weights()
			best_epochs = e

		if counter >= patience:
			model.set_weights(best_weights)
			print(f'Best epochs: {best_epochs}')
			break

	return model


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
def forecast_gaussian(model, predict_data, seq_len=50, forward=10, stride=1):
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

	return (np.concatenate(y_true, axis=1),
			np.concatenate(y_pred, axis=1),
			np.concatenate(y_var, axis=1))


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
	batch_size=10)


# dummy example
M = 2
x = np.random.normal(0., 1., size=(100, M)).astype('float32')
y = np.atleast_2d(x.sum(axis=-1) + np.random.normal(0., 0.1, size=100).astype('float32')).T

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

