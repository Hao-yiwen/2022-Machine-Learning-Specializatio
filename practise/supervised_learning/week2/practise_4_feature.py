import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng

np.set_printoptions(precision=2)

x = np.arange(0, 20, 1)
y = 1 +x**2
X = x.reshape(-1, 1)

model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-2)

plt.scatter(x,y,marker='x',c='r',label='真实值')
plt.title('真实值')
plt.plot(x,X@model_w+model_b, label="预测的值")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()

x = np.arange(0, 20, 1)
y = 1 +x**2
X = x**2
X = X.reshape(-1, 1)
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]
print(X)

model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-7)
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

print('--------')

x = np.arange(0,20,1)
y = x**2

X=np.c_[x, x**2, x**3]
X_features = ['x', 'x^2', 'x^3']

fig,ax=plt.subplots(1,3,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel('y')
plt.show()

x = np.arange(0,20,1)
X=np.c_[x,x**2,x**3]
print(np.ptp(X,axis=0))

X=zscore_normalize_features(X)
print(np.ptp(X,axis=0))

x=np.arange(0,20,1)
y=x**2

X=np.c_[x,x**2,x**3]
X=zscore_normalize_features(X)
model_w,model_b=run_gradient_descent_feng(X,y,iterations=10000,alpha=1e-1)
plt.scatter(x,y,marker='x',c='r',label='Actual Value')
plt.title('Normalized features')
plt.plot(x, np.dot(X,model_w)+model_b,label='Predicted Value')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
print(model_w, model_b)

print('------------')
x = np.arange(0,20,1)
y=np.cos(x/2)

X = np.c_[x,x**2,x**3,x**4,x**5,x**6,x**7,x**8,x**9,x**10,x**11,x**12,x**13,x**14,x**15]
X=zscore_normalize_features(X)

model_w,model_b=run_gradient_descent_feng(X,y,iterations=100000,alpha=1e-1)

plt.scatter(x,y,marker='x',c='r',label='Actual Value')
plt.title('Normalized features')
plt.plot(x, np.dot(X,model_w)+model_b,label='Predicted Value')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

if __name__ == '__main__':
    pass
