import numpy as np
import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh

input_array = np.array([1, 2, 3])
exp_array = np.exp(input_array)

print(f"input_array: {input_array}, exp_array: {exp_array}")

input_val = 1
exp_val = np.exp(input_val)

print(f"input_val: {input_val}, exp_val: {exp_val}")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

z_tmp = np.arange(-10, 11)

y = sigmoid(z_tmp)

np.set_printoptions(precision=3)
print(f"z_tmp: {z_tmp}, y: {y}")

fig,ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(z_tmp, y, c="b")
ax.set_xlabel('z')
ax.set_ylabel('y')
draw_vthresh(ax, 0)
plt.show()

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0

plt.close('all')
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)
plt.show()

if __name__ == '__main__':
    pass
