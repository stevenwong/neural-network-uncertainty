""" ts_uncertainty_train.py

Simulate neural network uncertainty.

Copyright (C) 2020 Steven Wong <steven.ykwong87@gmail.com>

MIT License

"""


import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# TF imports
import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose
from tensorflow.python.ops.gen_math_ops import mean
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
from tensorflow.keras.layers import Input, Lambda, Activation, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

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


def build_tcn_cov(seq_len,
				  input_dim,
				  output_dim,
				  dilation_rates,
				  filters=8,
				  kernel_size=5,
				  hidden_units=[8],
				  cov_factors=8,
				  cov_kernel=2,
				  cov_filters=4):
	nn = input_layer = Input(shape=(seq_len, input_dim))

	for i, d in enumerate(dilation_rates):
		nn = Conv1D(dilation_rate=d,
					filters=filters,
					kernel_size=kernel_size,
					padding='causal')(nn)
		nn = BatchNormalization()(nn)
		nn = Activation('relu')(nn)

	nn = latent = Lambda(lambda x: x[:, -1, :])(nn)

	# prediction layer
	for u in hidden_units:
		nn = Dense(u, activation='relu')(nn)

	output_layer = Dense(output_dim, activation='linear')(nn)

	# factor exposure and coviarance
	exposures = Dense(cov_factors, activation='linear')(latent)

	cov = K.expand_dims(latent)
	cov = K.expand_dims(cov)
	cov = Conv2DTranspose(cov_filters,
						  cov_kernel,
						  strides=(1, cov_factors),
						  padding='same',
						  activation='relu')(cov)
	cov = Conv2DTranspose(1,
						  cov_kernel,
						  padding='same')(cov)


	model = keras.Model(input_layer, [output_layer, exposures, cov])
	opt = Adam(lr=0.01)
	model.compile(loss='mse', optimizer=opt)

	return model, opt


# manually calculate MSE
def train_mse(model, x, y):
	y_ = model(x)
	return K.mean(K.square(y - y_), axis=0)


def test_mse(model, x, y):
	y_ = model.predict(x)
	return mean_squared_error(y, y_)


def train_mse_cov(model, x, y):
	y_, exposures, cov = model(x)
	e = y - y_
	with tf.stop_recording():
		S = e @ e.T
		L = tf.linalg.cholesky(S)
		mask = np.tril(np.ones_like(L))

	return K.mean(K.square(e), axis=0) + K.mean(K.square(exposures @ L - exposures @ (mask * cov)))


def calc_gradient(model, x, y, loss_fn):
	with tf.GradientTape() as tape:
		loss = loss_fn(model, x, y)
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
		val_loss = test_criterion(model, test_X, test_y)

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
	train_mse,
	test_mse,
	train_X,
	train_y,
	test_X,
	test_y)


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


y_true, y_pred = forecast(model, predict_data, seq_len=seq_len, forward=forward, stride=1)

x_axis = np.arange(y_true.shape[1])
fig, ax = plt.subplots()
ax.plot(x_axis, y_true[0,:], label='true')
ax.plot(x_axis, y_pred[0,:], label='pred')
plt.tight_layout()
plt.show()
plt.clf()


# build covariance network
model, opt = build_tcn_cov(seq_len,
	input_dim,
	input_dim,
	[1, 2, 4, 8],
	filters=8,
	kernel_size=5,
	hidden_units=[8],
	cov_factors=8,
	cov_kernel=2,
	cov_filters=4)


