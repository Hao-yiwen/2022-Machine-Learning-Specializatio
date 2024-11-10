import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])
y_train = np.array([300, 500])

print(f"x_train: {x_train}")
print(f"y_train: {y_train}")

print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"m: {m}")
m = len(x_train)
print(f"m: {m}")

i=0
x_i = x_train[i]
y_i = y_train[i]
print(f"x_{i}, y_{i}: {x_i}, {y_i}")

print(x_train, y_train)
plt.scatter(x_train, y_train, marker='x', c="r")
plt.title('房价')
plt.ylabel('房价 (以千美元为单位)')
plt.xlabel('房屋面积 (以平方英尺为单位)')
plt.show()

w=100
b=100
print(f"w: {w}, b: {b}")

def compute_model_output(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b)
print(x_train, w, b)

plt.plot(x_train, tmp_f_wb, c='b', label="our prediction")
plt.scatter(x_train, y_train, marker='x', c='r', label="training data")
plt.title('房价')
plt.ylabel('房价 (以千美元为单位)')
plt.xlabel('房屋面积 (以平方英尺为单位)')
plt.legend()
plt.show()


if __name__ == '__main__':
    pass