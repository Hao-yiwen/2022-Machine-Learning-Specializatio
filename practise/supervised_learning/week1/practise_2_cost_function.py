import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_cost(x, y, w, b):
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w*x[i] +b
        cost = (f_wb - y[i])**2
        cost_sum += cost
    total_cost = (1/(2*m)) * cost_sum

    return total_cost

plt_intuition(x_train,y_train)

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])
plt.close('all')

fig,ax,dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, dyn_items, x_train, y_train)

soup_bowl()

if __name__ == '__main__':
    pass