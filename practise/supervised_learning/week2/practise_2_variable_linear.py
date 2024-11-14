import copy, numpy as np, math
import matplotlib.pylab as plt

plt.style.use('./deeplearning.mplstyle')
# 设置numpy的打印选项，精度为2
np.set_printoptions(precision=2)

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

print(f"X_train: {X_train} X_train.shape: {X_train.shape}")
print(f"y_train: {y_train} y_train.shape: {y_train.shape}")

b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")


def predict_single_loop(x, w, b):
    n = x.shape[0]
    p = 0
    for i in range(n):
        p += x[i] * w[i]
    p += b
    return p


x_vec = X_train[0, :]
print(f"x_vec: {x_vec} x_vec.shape: {x_vec.shape}")

f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb: {f_wb}")


def predict(x, w, b):
    return np.dot(x, w) + b


x_vec = X_train[0, :]
f_wb = predict(x_vec, w_init, b_init)
print(f"f_wb: {f_wb}")


def compute_cost(X, y, w, b):
    # r = np.dot(X, w) + b - y
    p = 0
    n = X.shape[0]
    for i in range(n):
        p += (np.dot(X[i], w) + b - y[i]) ** 2
    return p / (2 * n)


cost = compute_cost(X_train, y_train, w_init, b_init)
print(f"cost: {cost}")


# print(f'X[i, j] ${X_train[1, 2] }')

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        err = np.dot(X[i], w) + b - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err
    return dj_dw / m, dj_db / m


dj_dw, dj_db = compute_gradient(X_train, y_train, w_init, b_init)
print(f"dj_dw: {dj_dw} dj_db: {dj_db}")


# X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
# y_train = np.array([460, 232, 178])
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b_in)
        w -= alpha * dj_dw
        b_in -= alpha * dj_db

        if i < 100000:  # 防止资源枯竭
            J_history.append(compute_cost(X, y, w, b_in))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i} cost: {J_history[-1]}")

    return w, b_in, J_history

init_w = np.zeros((X_train.shape[1],))
init_b = 0
iterations = 10000
alpha = 5e-7
final_w, final_b, J_history = gradient_descent(X_train, y_train, init_w, init_b, alpha, iterations)
print(f"final_w: {final_w} final_b: {final_b}")

for i in range(X_train.shape[0]):
    print(f"Predicted: {predict(X_train[i], final_w, final_b)} Actual: {y_train[i]}")

# 画图
fig, (ax1, ax2) = plt.subplots(1,2,constrained_layout=True, figsize=(10, 5))
ax1.plot(J_history)
ax2.plot(200 + np.arange(len(J_history[200:])), J_history[200:])

ax1.set_title('Cost function')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Cost')

ax2.set_title('Cost function after 200')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost')

plt.show()

if __name__ == '__main__':
    pass
