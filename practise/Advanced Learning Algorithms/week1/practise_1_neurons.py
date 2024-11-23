import logging

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from lab_utils_common import dlc
from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic

plt.style.use('./deeplearning.mplstyle')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X_train = np.array([[1.0], [2.0]], dtype=np.float32)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)

fig, ax = plt.subplots(1, 1)
ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
ax.legend(fontsize="xx-large")
ax.set_ylabel("Y")
ax.set_xlabel("X")
plt.show()

linear_layer = tf.keras.layers.Dense(units=1, activation='linear')

print(linear_layer.get_weights())

a1 = linear_layer(X_train[0].reshape(1, 1))
print(a1)

w, b = linear_layer.get_weights()
print(w, b)

set_w = np.array([[300]])
set_b = np.array([100])

linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())

a1 = linear_layer(X_train[0].reshape(1, 1))
print(a1)
alin = np.dot(set_w, X_train[0].reshape(1, 1)) + set_b
print(alin)

prediction_tf = linear_layer(X_train)
prediction_np = np.dot(X_train, set_w) + set_b
print("==========")
print(prediction_tf, prediction_np)

plt_linear(X_train, Y_train, prediction_tf, prediction_np)

X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1, 1)
Y_train = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32).reshape(-1, 1)
pos = Y_train == 1
neg = Y_train == 0
print('===========')
print(neg, pos)
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.scatter(X_train[pos], Y_train[pos], c='r', label='Positive')
ax.scatter(X_train[neg], Y_train[neg], c='b', label='Negative')

ax.set_ylim(-0.08, 1.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend(fontsize='xx-large')
plt.show()

model = Sequential(
    [
        tf.keras.layers.Dense(1, input_dim=1, activation='sigmoid', name='L1')
    ]
)

model.summary()

logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()
print(w, b)

set_w = np.array([[2]])
set_b = np.array([-4.5])
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())

a1 = model.predict(X_train[0].reshape(1, 1))
print(a1)
alog = sigmoidnp(np.dot(set_w, X_train[0].reshape(1, 1)) + set_b)
print(alog)

plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg)

if __name__ == '__main__':
    pass
