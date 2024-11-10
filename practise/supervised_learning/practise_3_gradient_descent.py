import math, copy
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')

from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2*m) * cost

    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = 1 / m * dj_dw
    dj_db = 1 / m * dj_db

    return dj_dw, dj_db

plt_gradients(x_train, y_train, compute_cost, compute_gradient)
plt.show()

if __name__ == '__main__':
    pass