import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import dlc, plot_data
from plt_one_addpt_onclick import plt_one_addpt_onclick

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])

pos = y_train == 1
neg = y_train == 0
print(f"pos: {pos}, neg: {neg}")

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(x_train[pos], y_train[pos], color='red', marker='o', label='Positive')
ax[0].scatter(x_train[neg], y_train[neg], color='blue', marker='x', label='Negative')

ax[0].set_xlabel('X')
ax[0].set_ylabel('y')
ax[0].legend(loc='upper left')

plot_data(X_train2, y_train2, ax[1])
ax[1].axis([0, 5, 0, 5])
ax[1].set_xlabel('X')
ax[1].set_ylabel('y')
ax[1].legend(loc='upper left')

plt.tight_layout()
plt.show()

import matplotlib

print(matplotlib.__version__)

w_in = np.zeros((1))
b_in = 0
plt.close('all')
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=False)

if __name__ == '__main__':
    pass