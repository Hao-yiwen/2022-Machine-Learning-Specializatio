import numpy as np
from IPython.core.pylabtools import figsize

np.set_printoptions(precision=2)
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import load_house_data
import matplotlib.pyplot as plt

dlblue = '#0096ff';
dlorange = '#FF9300';
dldarkred = '#C00000';
dlmagenta = '#FF40FF';
dlpurple = '#7030A0';
plt.style.use('./deeplearning.mplstyle')

X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(np.ptp(X_train))
print(np.ptp(X_norm))

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(b_norm, w_norm)
print(f"model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

y_pred_sgd = sgdr.predict(X_norm)
y_pred = np.dot(X_norm, w_norm) + b_norm

print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}")
print(f"Target values \n{y_train[:4]}")

fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train)
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:, i], y_pred, c='r', label='Predicted')
ax[0].set_ylabel('Price')
plt.show()

if __name__ == '__main__':
    pass
