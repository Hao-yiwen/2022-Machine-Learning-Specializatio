import numpy as np

# 设置numpy的打印选项，精度为2，即小数点后保留2位
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt

dlblue = '#375E97'
dlorange = '#FFA600'
dldarked = '#3F681C'
dlmagenta = '#F1646E'
dlpurple = '#7D8CC4'
plt.style.use('./deeplearning.mplstyle')
from lab_utils_multi import load_house_data, compute_cost, run_gradient_descent
from lab_utils_multi import norm_plot, plt_contour_multi, plt_equal_scale, plot_cost_i_w

X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

fig, ax = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, color=dlblue)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel('Price')
plt.show()


def zscore_normalize_features(X):
    # axis=0 表示沿着列方向计算均值，即对每一列/特征求平均值
    # 例如对于一个m x n的矩阵，结果是一个长度为n的向量，每个元素是对应列的均值
    mu = np.mean(X, axis=0)
    # axis=0 表示沿着列方向计算标准差，即对每一列/特征求标准差
    # 例如对于一个m x n的矩阵，结果是一个长度为n的向量，每个元素是对应列的标准差
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)


# from sklearn.preprocessing import scale
# scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)

mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)
X_mean = (X_train - mu)
X_zscore = (X_train - mu) / sigma  # zscore标准化后的特征矩阵，每个特征都减去均值并除以标准差

fig, ax = plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:, 0], X_train[:, 3], color=dlblue)
ax[0].set_xlabel(X_features[0])
ax[0].set_ylabel(X_features[3])
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_mean[:, 0], X_mean[:, 3])
ax[1].set_xlabel(X_features[0]);
ax[0].set_ylabel(X_features[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_zscore[:, 0], X_zscore[:, 3])
ax[2].set_xlabel(X_features[0]);
ax[2].set_ylabel(X_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")

plt.show()

X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print("X_norm[:5] = \n", X_norm)
print(X_mu)
print(X_sigma)
# np.ptp(X_train, axis=0) 计算每一列的极差(peak to peak)
# axis=0 表示沿着列方向计算，即对每一列/特征求极差
# 极差 = 最大值 - 最小值
# 返回一个长度为n的向量，每个元素是对应列的极差
print(np.ptp(X_train, axis=0))
print(np.ptp(X_norm, axis=0))

fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i], X_train[:, i])
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel('count')
fig.suptitle('归一化之前的特征分布')
plt.show()
fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i], X_norm[:, i])
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel('count')
fig.suptitle('归一化之前的特征分布')
plt.show()

w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )

m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, color=dlblue)
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:, i], yp, color=dlorange)
ax[0].set_ylabel('Price')
ax[0].legend(['actual', 'predicted'])
plt.show()

x_house = np.array([1200,3,1,40])
x_house_norm = (x_house-X_mu)/X_sigma
print("x_house_norm = ", x_house_norm)
x_house_pred = np.dot(x_house_norm, w_norm) + b_norm
print("x_house_pred = ", x_house_pred)
plt_equal_scale(X_train, X_norm, y_train)

if __name__ == '__main__':
    pass
