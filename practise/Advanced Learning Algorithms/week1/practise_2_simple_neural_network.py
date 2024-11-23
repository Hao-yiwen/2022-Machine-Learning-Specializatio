import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
X, Y = load_coffee_data()
print(X.shape, Y.shape)

plt_roast(X, Y)

print(f"Temperature Max, Min pre normalization: {np.max(X[:, 0]):0.2f}, {np.min(X[:, 0]):0.2f}")
print(f"Duration Max, Min pre normalization: {np.max(X[:, 1]):0.2f}, {np.min(X[:, 1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:, 0]):0.2f}, {np.min(Xn[:, 0]):0.2f}")
print(f"Duration Max, Min post normalization: {np.max(Xn[:, 1]):0.2f}, {np.min(Xn[:, 1]):0.2f}")

Xt = np.tile(Xn, (1000, 1))
yt = np.tile(Y, (1000, 1))
print(Xt.shape, yt.shape)

tf.random.set_seed(1234)

model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name="layer1"),
        Dense(1, activation='sigmoid', name="layer2")
    ]
)

model.summary()

L1_num_params = 2 * 3 + 3
L2_num_params = 3 * 1 + 1
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              )

model.fit(Xt, yt, epochs=10, verbose=1)

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

X_test = np.array([
    [200, 13.9],  # postive example
    [200, 17] # negative example
]);
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print(f"Predictions: {predictions}")



if __name__ == '__main__':
    pass
