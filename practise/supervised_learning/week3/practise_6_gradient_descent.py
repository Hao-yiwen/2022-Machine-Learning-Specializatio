import copy, math
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import dlc, plot_data, plt_tumor_data, sigmoid, compute_cost_logistic
from plt_quad_logistic import plt_quad_logistic, plt_prob
from practise.supervised_learning.week2.practise_2_variable_linear import dj_dw
from practise.supervised_learning.week3.practise_5_cost_function import b_tmp

plt.style.use('./deeplearning.mplstyle')

X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plot_data(X_train, y_train, ax)
ax.axis([0, 4, 0, 3])
plt.show()


def compute_gradient_logistic(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


X_tmp = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2., 3.])
b_tmp = 1

dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}")
print(f"dj_dw: {dj_dw_tmp.tolist()}")


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 10000:
            J_history.append(compute_cost_logistic(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i}, cost: {J_history[-1]:.2f}")

    return w, b, J_history


w_tmp = np.zeros_like(X_train[0])
b_tmp = 0
alpha = 0.1
num_iters = 10000

w, b, J_history = gradient_descent(X_train, y_train, w_tmp, b_tmp, alpha, num_iters)
print(f"w: {w}")
print(f"b: {b}")

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plt_prob(ax, w, b)
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
ax.axis([0, 4, 0, 3])
plot_data(X_train, y_train, ax)

x0 = -b / w[0]
x1 = -b / w[1]
ax.plot([0,x0],[x1,0], c=dlc["dlblue"], lw=1)
plt.show()

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

fig,ax = plt.subplots(1,1,figsize=(4,3))
plt_tumor_data(x_train, y_train, ax)
plt.show()

w_range = np.array([-1, 7])
b_range = np.array([1, -11])
quad = plt_quad_logistic(x_train, y_train, w_range, b_range)

plt.show()

if __name__ == '__main__':
    pass
