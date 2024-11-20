import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import plot_data, sigmoid, draw_vthresh

plt.style.use('./deeplearning.mplstyle')

X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)

print("X.shape:", X, X.shape)
print("y.shape:", y, y.shape)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plot_data(X, y, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()

z = np.arange(-10, 11)
fig,ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(z, sigmoid(z), label='sigmoid')

ax.set_title('Sigmoid function')
ax.set_xlabel('z')
ax.set_ylabel('$\sigma(z)$')
draw_vthresh(ax, 0)
plt.show()

x0= np.arange(0,6)
x1= 3-x0
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)

# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()

if __name__ == "__main__":
    pass
