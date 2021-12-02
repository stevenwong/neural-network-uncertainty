""" network.py

Various neural network uncertainty layers.

Copyright (C) 2021 Steven Wong <steven.ykwong87@gmail.com>

MIT License

"""

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import GlorotNormal

import numpy as np
import pandas as pd
import pickle
import math
import multiprocessing
import itertools
import gc

from collections import Iterable
from sklearn.metrics import mean_squared_error


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
		# sigmoid transform only works with NLL loss
		output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06
		return [output_mu, output_sig_pos]

	def compute_output_shape(self, input_shape):
		return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]


class MultivariateGaussian(Layer):
	def __init__(self, cov_dim, output_dim, **kwargs):
		self.cov_dim = cov_dim
		self.output_dim = output_dim
		self.input_dim = None
		super(MultivariateGaussian, self).__init__(**kwargs)

	def build(self, input_shape):
		input_dim = input_shape[-1]
		# mu weights
		self.mu_kernel = self.add_weight(name='mu_kernel', 
										 shape=(input_dim, self.output_dim),
										 initializer=GlorotNormal(),
										 trainable=True)
		# correlation scaling kernel
		self.rho_kernel = self.add_weight(name='rho_kernel',
										  shape=(self.output_dim, ),
										  initializer=GlorotNormal(),
										  trainable=True)
		# Convert latent representation to covariance space
		# self.cov_kernel = self.add_weight(name='cov_kernel',
		# 								  shape=(input_dim, self.cov_dim),
		# 								  initializer=GlorotNormal(),
		# 								  trainable=True)
		# variance scaling kernel
		self.var_kernel = self.add_weight(name='var_kernel',
										  shape=(input_dim, self.output_dim),
										  initializer=GlorotNormal(),
										  trainable=True)
		# scaling
		# self.scale_kernel = self.add_weight(name='scale_kernel',
		# 									shape=(self.output_dim, ),
		# 									initializer=GlorotNormal(),
		# 									trainable=True)
		# mu bias
		self.mu_bias = self.add_weight(name='mu_bias',
									   shape=(self.output_dim, ),
									   initializer=GlorotNormal(),
									   trainable=True)
		# variance bias
		self.var_bias = self.add_weight(name='var_bias',
										shape=(self.output_dim, ),
										initializer=GlorotNormal(),
										trainable=True)
		# scale bias
		# self.scale_bias = self.add_weight(name='scale_bias',
		# 								  shape=(self.output_dim, ),
		# 								  initializer=GlorotNormal(),
		# 								  trainable=True)
		# covariance bias
		# self.cov_bias = self.add_weight(name='cov_bias',
		# 								shape=(self.cov_dim, ),
		# 								initializer=GlorotNormal(),
		# 								trainable=True)
		# number of entries in lower triangular matrix is N(N + 1) / 2
		# ones = tf.ones(int(self.cov_dim * (self.cov_dim + 1) / 2))
		# self.mask = tfp.math.fill_triangular(ones)
		self.input_dim = input_dim
		super(MultivariateGaussian, self).build(input_shape) 

	def call(self, x):
		# mean estimate
		n = tf.shape(x)[0]
		output_mu  = K.dot(x, self.mu_kernel) + self.mu_bias

		# x has shape (n, input_dim)
		# permutations of x. Has shape (n, n, input_dim)
		# c = tf.tanh(K.dot(x, self.cov_kernel) + self.cov_bias)
		a = tf.expand_dims(x, axis=1)	# shape = (n, 1, input_dim)
		a = tf.tile(a, (1, n, 1))
		# var_a = K.sum(a * tf.squeeze(self.var_kernel), axis=-1) + self.var_bias

		b = tf.expand_dims(x, axis=0)
		b = tf.tile(b, (n, 1, 1))
		# var_b = K.sum(b * tf.squeeze(self.var_kernel), axis=-1) + self.var_bias

		d = K.dot(x, self.var_kernel) + self.var_bias
		d = K.log(1 + K.exp(d)) + 1e-08
		d = tf.squeeze(d)
		# d = tf.linalg.diag(d)

		# scale = tf.concat([a, b], axis=2)
		# scale = K.sum(a * b * self.scale_kernel, axis=-1) + self.scale_bias
		# scale = K.log(1 + K.exp(scale)) + 1e-08
		# scale = tf.square(scale)

		rho = K.sum(a * b * self.rho_kernel, axis=-1)	# / (tf.norm(a, axis=-1) * tf.norm(b, axis=-1))
		# rho = tf.tanh(K.sum(var_a * var_b, axis=-1))
		# rho = tf.concat([a, b], axis=2)
		# rho = tf.tanh(K.sum(rho * self.rho_kernel, axis=-1))
		# cov = rho + d
		cov = tf.linalg.set_diag(rho, d)
		output_cov = tf.squeeze(cov)

		return [output_mu, output_cov]

	def compute_output_shape(self, input_shape):
		return [(input_shape[0], self.output_dim), (input_shape[0], input_shape[0])]


class AlternativeMVN(Layer):
	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		self.input_dim = None
		super(AlternativeMVN, self).__init__(**kwargs)

	def build(self, input_shape):
		input_dim = input_shape[-1]
		# mu weights
		self.mu_kernel = self.add_weight(name='mu_kernel', 
										 shape=(input_dim, self.output_dim),
										 initializer=GlorotNormal(),
										 trainable=True)
		# correlation scaling kernel
		self.rho_kernel = self.add_weight(name='rho_kernel',
										  shape=(self.output_dim, ),
										  initializer=GlorotNormal(),
										  trainable=True)
		# Convert latent representation to covariance space
		self.cov_kernel = self.add_weight(name='cov_kernel',
										  shape=(input_dim, self.cov_dim),
										  initializer=GlorotNormal(),
										  trainable=True)
		# variance scaling kernel
		self.var_kernel = self.add_weight(name='var_kernel',
										  shape=(input_dim, self.output_dim),
										  initializer=GlorotNormal(),
										  trainable=True)
		# scaling
		# self.scale_kernel = self.add_weight(name='scale_kernel',
		# 									shape=(self.output_dim, ),
		# 									initializer=GlorotNormal(),
		# 									trainable=True)
		# mu bias
		self.mu_bias = self.add_weight(name='mu_bias',
									   shape=(self.output_dim, ),
									   initializer=GlorotNormal(),
									   trainable=True)
		# variance bias
		self.var_bias = self.add_weight(name='var_bias',
										shape=(self.output_dim, ),
										initializer=GlorotNormal(),
										trainable=True)
		# scale bias
		# self.scale_bias = self.add_weight(name='scale_bias',
		# 								  shape=(self.output_dim, ),
		# 								  initializer=GlorotNormal(),
		# 								  trainable=True)
		# covariance bias
		self.cov_bias = self.add_weight(name='cov_bias',
										shape=(self.cov_dim, ),
										initializer=GlorotNormal(),
										trainable=True)
		# number of entries in lower triangular matrix is N(N + 1) / 2
		ones = tf.ones(int(self.cov_dim * (self.cov_dim + 1) / 2))
		self.mask = tfp.math.fill_triangular(ones)
		self.input_dim = input_dim
		super(MultivariateGaussian, self).build(input_shape) 

	def call(self, x):
		# mean estimate
		n = tf.shape(x)[0]
		output_mu  = K.dot(x, self.mu_kernel) + self.mu_bias

		# x has shape (n, input_dim)
		# permutations of x. Has shape (n, n, input_dim)
		c = K.dot(x, self.cov_kernel) + self.cov_bias
		a = tf.expand_dims(c, axis=1)	# shape = (n, 1, input_dim)
		a = tf.tile(a, (1, n, 1))
		# var_a = K.sum(a * tf.squeeze(self.var_kernel), axis=-1) + self.var_bias

		b = tf.expand_dims(c, axis=0)
		b = tf.tile(b, (n, 1, 1))
		# var_b = K.sum(b * tf.squeeze(self.var_kernel), axis=-1) + self.var_bias

		d = K.dot(x, self.var_kernel) + self.var_bias
		d = K.log(1 + K.exp(d)) + 1e-08
		d = tf.squeeze(d)
		# d = tf.linalg.diag(d)

		# scale = tf.concat([a, b], axis=2)
		# scale = K.sum(a * b * self.scale_kernel, axis=-1) + self.scale_bias
		# scale = K.log(1 + K.exp(scale)) + 1e-08
		# scale = tf.square(scale)

		# rho = K.sum(a * b * self.rho_kernel, axis=-1)	# / (tf.norm(a, axis=-1) * tf.norm(b, axis=-1))
		# rho = tf.tanh(K.sum(var_a * var_b, axis=-1))
		rho = tf.concat([a, b], axis=2)
		rho = K.sum(rho * self.rho_kernel, axis=-1)
		rho = rho * self.mask
		rho = tf.matmul(rho, tf.transpose(rho))
		# cov = rho + d
		cov = tf.linalg.set_diag(rho, d)
		output_cov = tf.squeeze(cov)

		return [output_mu, output_cov]

	def compute_output_shape(self, input_shape):
		return [(input_shape[0], self.output_dim), (input_shape[0], input_shape[0])]


class CovarianceMatrix(Layer):
	def __init__(self, output_dim=1, **kwargs):
		self.output_dim = output_dim
		self.input_dim = None
		super(CovarianceMatrix, self).__init__(**kwargs)

	def build(self, input_shape):
		input_dim = input_shape[-1]
		# correlation scaling kernel
		# self.rho_kernel = self.add_weight(name='rho_kernel',
		# 								  shape=(input_dim, ),
		# 								  initializer=GlorotNormal(),
		# 								  trainable=True)
		# Convert latent representation to covariance space
		self.cov_kernel = self.add_weight(name='cov_kernel',
										  shape=(input_dim, ),
										  initializer=GlorotNormal(),
										  trainable=True)
		# variance scaling kernel
		self.var_kernel = self.add_weight(name='var_kernel',
										  shape=(input_dim, self.output_dim),
										  initializer=GlorotNormal(),
										  trainable=True)
		# scaling
		# self.scale_kernel = self.add_weight(name='scale_kernel',
		# 									shape=(input_dim, ),
		# 									initializer=GlorotNormal(),
		# 									trainable=True)
		# variance bias
		self.var_bias = self.add_weight(name='var_bias',
										shape=(self.output_dim, ),
										initializer=GlorotNormal(),
										trainable=True)
		# scale bias
		# self.scale_bias = self.add_weight(name='scale_bias',
		# 								  shape=(self.output_dim, ),
		# 								  initializer=GlorotNormal(),
		# 								  trainable=True)
		# covariance bias
		# self.cov_bias = self.add_weight(name='cov_bias',
		# 								shape=(self.output_dim, ),
		# 								initializer=GlorotNormal(),
		# 								trainable=True)
		self.input_dim = input_dim
		super(CovarianceMatrix, self).build(input_shape) 

	def call(self, x):
		# mean estimate
		n = tf.shape(x)[0]
		# x has shape (n, input_dim)
		# permutations of x. Has shape (n, n, input_dim)
		a = tf.expand_dims(x, axis=1)	# shape = (n, 1, input_dim)
		a = tf.tile(a, (1, n, 1))
		# var_a = K.sum(a * tf.squeeze(self.var_kernel), axis=-1) + self.var_bias

		b = tf.expand_dims(x, axis=0)
		b = tf.tile(b, (n, 1, 1))
		# var_b = K.sum(b * tf.squeeze(self.var_kernel), axis=-1) + self.var_bias

		d = K.dot(x, self.var_kernel) + self.var_bias
		d = K.log(1 + K.exp(d)) + 1e-08
		# d = tf.square(d)
		d = tf.squeeze(d)
		# d = tf.linalg.diag(d)

		# scale = tf.concat([a, b], axis=2)
		# scale = K.sum(a * b * self.scale_kernel, axis=-1) + self.scale_bias
		# scale = K.log(1 + K.exp(scale)) + 1e-08
		# scale = tf.square(scale)

		rho = K.sum(a * b * self.cov_kernel, axis=-1) # / (tf.norm(a, axis=-1) * tf.norm(b, axis=-1))
		# rho = tf.concat([a, b], axis=2)
		# rho = tf.tanh(K.sum(rho * self.rho_kernel, axis=-1))
		# cov = rho + d
		cov = tf.linalg.set_diag(rho, d)
		output_cov = tf.squeeze(cov)

		return output_cov

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[0])


def mse(y_true, y_pred, tape=None):
	return tf.reduce_mean(tf.square(y_pred - y_true))


# negative log-likelihood multivariate normal
def nll(y, out, tape=None):
	u, v = out
	# NLL = log(var(x))/2 + (y - u(x))^2 / 2var(x)
	return tf.reduce_mean(0.5 * tf.math.log(v) +
		0.5 * tf.math.divide(tf.math.square(y - u), v))


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
			# print(f'|e| {tf.reduce_mean(tf.square(e))} |v| {tf.reduce_mean(tf.square(V - v))}')
	else:
		V = K.dot(e, K.transpose(e))
		# V = tf.square(e)
	return tf.reduce_mean(tf.square(e)) + tf.reduce_mean(tf.square(V - v))


def var_mse(y, out, tape=None):
	# see `https://github.com/sthorn/deep-learning-explorations/blob/master/predicting-uncertainty-variance.ipynb`
	u, v = out
	e = y - u
	if tape is not None:
		with tape.stop_recording():
			V = K.dot(y, K.transpose(y))
			# print(f'|e| {tf.reduce_mean(tf.square(e))} |v| {tf.reduce_mean(tf.square(V - v))}')
	else:
		V = K.dot(e, K.transpose(e))
	return tf.reduce_mean(tf.square(e)) + tf.reduce_mean(tf.square(V - v))


def calc_gradient(model, x, y, loss_fn):
	with tf.GradientTape() as tape:
		y_ = model(x)
		loss = loss_fn(y, y_, tape=tape)
		grads = tape.gradient(loss, model.trainable_variables)
		if tf.math.is_nan(grads[-1][-1]):
			print('grads', grads)
			print('loss', loss)
			print('Dumping input to data.pkl')
			pickle.dump((x, y), open('data.pkl', 'wb'))
			raise ValueError('Invalid gradient found')
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
				patience=10,
				hp_mode=False):
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
			batch_X = tf.convert_to_tensor(train_X[idx])
			batch_y = tf.convert_to_tensor(train_y[idx])

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

	if hp_mode:
		return best_loss
	else:
		return model


def hyperparameter_search(hyperparams, train_fn, *args, n_cpu=2, **kwargs):
	""" Given a function that creates estimators, training and validation data, search through all
	combinations of hyperparameters.

	Args:
		hyperparams (dict): Dictionary with hyperparameters.
		estimator_fn (function): Function which accepts **kwargs and returns an estimator.
		train_fn (function): Function used to train network and returns predictions
		n_cpu (int): Number of parallel processes to run.
		*args: Arguments for `train_fn`.
		**kwargs: Keyword arguments for `train_fn`.

	Returns:
		(dict, ): Best hyperparameters.

	"""

	results = []

	keys = []
	permutations = []

	for k in hyperparams:
		keys.append(k)
		v = hyperparams[k]

		if not isinstance(v, Iterable):
			permutations.append([v])
		else:
			permutations.append(v)

	permutations = list(itertools.product(*permutations))
	n = len(permutations)
	scores = []
	best_val = np.inf
	best_params = None

	if n_cpu > 1:
		with multiprocessing.Pool(processes=min(n, n_cpu)) as pool:
			for i in range(n):
				params = dict(zip(keys, permutations[i]))
				print('Testing iteration {} params {}'.format(i, params))

				hdl = pool.apply_async(train_fn, args=(params, *args), kwds=kwargs)

				results.append((hdl, params))

			for hdl, params in results:
				val_value = hdl.get()
				scores.append((params, val_value))

				if val_value < best_val:
					best_val = val_value
					best_params = params

	else:
		for i in range(n):
			params = dict(zip(keys, permutations[i]))
			print('Testing iteration {} params {}'.format(i, params))
			val_value = train_fn(params, *args, **kwargs)
			scores.append((params, val_value))

			if val_value < best_val:
				best_val = val_value
				best_params = params

			gc.collect()

	print('Best params {} loss {:4f}'.format(best_params, val_value))

	return best_params, scores


def forecast_error(y_true, y_pred, var_type='error'):
	T, N, M = y_true.shape
	covs = []

	for i in range(T):
		if var_type == 'error':
			e = y_true[i] - y_pred[i]
			covs.append(e @ e.T)
		elif var_type == 'y':
			covs.append(y_true[i] @ y_true[i].T)

	return np.stack(covs, axis=0)


def score(y_true, y_pred, cov_true, cov_pred):
	T, N, M = y_true.shape

	y_corr = np.zeros(T)
	diag_corr = np.zeros(T)
	tril_corr = np.zeros(T)
	y_mse = np.zeros(T)
	diag_mse = np.zeros(T)
	tril_mse = np.zeros(T)
	for i in range(T):
		y_corr[i] = np.corrcoef(y_true[i], y_pred[i], rowvar=False)[0,1]
		diag_corr[i] = np.corrcoef(np.diag(cov_true[i]), np.diag(cov_pred[i]), rowvar=False)[0,1]
		tril_corr[i] = np.corrcoef(np.tril(cov_true[i]).ravel(), np.tril(cov_pred[i]).ravel(), rowvar=False)[0,1]
		y_mse[i] = mean_squared_error(y_true[i], y_pred[i])
		diag_mse[i] = mean_squared_error(np.diag(cov_true[i]), np.diag(cov_pred[i]))
		tril_mse[i] = mean_squared_error(np.tril(cov_true[i]), np.tril(cov_pred[i]))

	return pd.Series({
		'y corr': y_corr.mean(),
		'diag corr': diag_corr.mean(),
		'tril corr': tril_corr.mean(),
		'y mse': y_mse.mean(),
		'diag mse': diag_mse.mean(),
		'tril mse': tril_mse.mean()
	})

