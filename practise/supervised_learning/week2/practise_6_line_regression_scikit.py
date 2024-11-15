import numpy as np

np.set_printoptions(precision=2)
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import load_house_data
import matplotlib.pyplot as plt
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0';
plt.style.use('./deeplearning.mplstyle')

X_train = np.array([1.0, 2.0])
y_train = np.array([2.0, 3.0])

linear_model = LinearRegression()
linear_model.fit(X_train.reshape(-1, 1), y_train)

b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
print(f"'manual' prediction: f_wb = wx+b : {1200*w + b}")

y_pred = linear_model.predict(X_train.reshape(-1, 1))
print(f"Prediction: {y_pred}")
X_test = np.array([[1200]])
y_pred = linear_model.predict(X_test)
print(f"Prediction: {y_pred}")

print('-------------------------')
X_train,y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

linear_model = LinearRegression()
linear_model.fit(X_train,y_train)
b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
print(f"Prediction on training set:\n {linear_model.predict(X_train)[:4]}" )
print(f"prediction using w,b:\n {(X_train @ w + b)[:4]}")
print(f"Target values \n {y_train[:4]}")

x_house = np.array([1200, 3,1, 40]).reshape(-1,4)
x_house_predict = linear_model.predict(x_house)[0]
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.2f}")

if __name__ == '__main__':
    pass