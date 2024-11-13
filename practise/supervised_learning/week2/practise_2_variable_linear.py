import copy,numpy as np, math
import matplotlib.pylab as plt

plt.style.use('./deeplearning.mplstyle')
# 设置numpy的打印选项，精度为2
np.set_printoptions(precision=2)

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

print(f"X_train: {X_train} X_train.shape: {X_train.shape}")
print(f"y_train: {y_train} y_train.shape: {y_train.shape}")

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")


def predict_single_loop(x, w, b):
    n = x.shape[0]
    p = 0
    for i in range(n):
        p += x[i] * w[i]
    p += b
    return p

x_vec = X_train[0,:]
print(f"x_vec: {x_vec} x_vec.shape: {x_vec.shape}")

f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb: {f_wb}")

def predict(x, w, b):
    return np.dot(x, w) + b

x_vec = X_train[0,:]
f_wb = predict(x_vec, w_init, b_init)
print(f"f_wb: {f_wb}")

def compute_cost(X, y ,w, b):
    # r = np.dot(X, w) + b - y
    p = 0
    n = X.shape[0]
    for i in range(n):
        p += (np.dot(X[i], w) + b - y[i])**2
    return p/(2*n)

cost = compute_cost(X_train, y_train, w_init, b_init)
print(f"cost: {cost}")

if __name__ == '__main__':
    pass