import numpy as np
import matplotlib.pyplot as plt

from lab_utils_common import plot_data, sigmoid, dlc
plt.style.use('./deeplearning.mplstyle')

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])

fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()

def compute_cost_logistic(X, y,w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)

    cost /= m
    return cost

w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))

x0 = np.arange(0,6)
x1 = 3- x0
x1_other = 4 - x0

fig,ax = plt.subplots(1,1, figsize=(4,4))
ax.plot(x0, x1, c= dlc['dlblue'], label='$b$=-3')
ax.plot(x0, x1_other, c=dlc['dlorange'], label='$b$=-4')
ax.axis([0,4,0,4])

plot_data(X_train, y_train, ax)
ax.axis([0,4,0,4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend()
plt.show()

w_array1 = np.array([1,1])
b_1  = -3
w_array2 = np.array([1,1])
b_2 = -4

cost1 = compute_cost_logistic(X_train, y_train, w_array1, b_1)
cost2 = compute_cost_logistic(X_train, y_train, w_array2, b_2)

print(f'Cost with b=-3: {cost1:.2f}')
print(f'Cost with b=-4: {cost2:.2f}')

if __name__ == "__main__":
    pass