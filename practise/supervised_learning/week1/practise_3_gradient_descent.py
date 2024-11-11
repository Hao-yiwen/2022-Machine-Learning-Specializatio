import math, copy
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('deeplearning.mplstyle')

from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost

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


def gradient_descent(x, y, w_in, b_in, alpha, num_iter, const_function, gradient_function):
    w = copy.deepcopy(w_in)
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iter):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        if i < 100000:
            J_history.append(const_function(x, y, w, b))
            p_history.append([w, b])

        if i % math.ceil(num_iter / 10) == 0:
            print(f'迭代次数 {i:4}: 损失函数 {J_history[-1]:0.2e}',
                  f"dj_dw:{dj_dw: 0.3e}, dj_db:{dj_db: 0.3e}",
                  f"w:{w: 0.3e}, b:{b: 0.5e}"
                  )

    return w, b, J_history, p_history


w_in = 0
b_in = 0
iterations = 40000
tmp_alpha = 1.0e-3
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_in, b_in, tmp_alpha, iterations, compute_cost,compute_gradient)

print(f'最终的w: {w_final:0.2f}, 最终的b: {b_final:0.2f}')

fig, (ax1, ax2) = plt.subplots(1,2, constrained_layout=True, figsize=(10, 5))
ax1.plot(J_hist[:1000])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()

if __name__ == '__main__':
    pass
