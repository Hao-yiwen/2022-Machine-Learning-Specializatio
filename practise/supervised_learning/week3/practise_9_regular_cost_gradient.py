import numpy as np
import matplotlib.pyplot as plt

from plt_overfit import overfit_example, output
from lab_utils_common import sigmoid

np.set_printoptions(precision=8)


def compute_cost_linear_reg(X, y, w, b, lambda_=1):
    m = X.shape[0]
    n = len(w)
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i]) ** 2
    cost /= (2 * m)

    reg_cost = 0
    for j in range(n):
        reg_cost += w[j] ** 2
    reg_cost = (lambda_ / (2 * m)) * reg_cost

    return cost + reg_cost


np.random.seed(1)
X_tmp = np.random.rand(5, 6)
print(X_tmp)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1, ) - 0.5
print(w_tmp)
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f'Cost with lambda={lambda_tmp:.2f}: {cost_tmp:.2f}')


def compute_cost_logistic_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost /= m

    reg_cost = 0
    for j in range(n):
        reg_cost += w[j] ** 2
    reg_cost = (lambda_ / (2 * m)) * reg_cost

    return cost + reg_cost


np.random.seed(1)
X_tmp = np.random.rand(5, 6)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1, ) - 0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f'Cost with lambda={lambda_tmp:.2f}: {cost_tmp:.2f}')

def compute_gradient_linear_reg(X, y, w, b, lambda_):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i][j]
        dj_db += err
    dj_dw /= m
    dj_db /= m

    for j in range(n):
        dj_dw[j] += (lambda_/m) * w[j]

    return dj_dw, dj_db

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)
b_tmp = 0.5
lambda_tmp = 0.7
dj_dw_tmp, dj_db_tmp = compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f'dJ/dw: {dj_dw_tmp}')
print(f'dJ/db: {dj_db_tmp}')

def compute_gradient_logistic_reg(X, y, w,b,lambda_=1):
    m, n = X.shape
    dj_dw=np.zeros((n,))
    dj_db=0.0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i][j]
        dj_db += err
    dj_dw /= m
    dj_db /= m

    for j in range(n):
        dj_dw[j] += (lambda_/m) * w[j]

    return dj_dw, dj_db

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

if __name__ == '__main__':
    pass;
