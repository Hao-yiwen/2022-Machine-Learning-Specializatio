import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 数据准备
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)

# 2. 数据预处理
X_b = np.hstack((np.ones((X.shape[0], 1)), X))
X_train, X_test, y_train, y_test = train_test_split(
    X_b, y, test_size=0.2, random_state=42
)

# 3. 定义模型

# 正规方程解
def normal_equation(X, y):
    theta_best = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta_best

# 梯度下降相关函数
def compute_loss(X, y, theta):
    m = len(y)
    predictions = X @ theta
    loss = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return loss

def compute_gradient(X, y, theta):
    m = len(y)
    predictions = X @ theta
    gradient = (1 / m) * X.T @ (predictions - y)
    return gradient

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    loss_history = []
    for i in range(num_iterations):
        gradient = compute_gradient(X, y, theta)
        theta -= learning_rate * gradient
        loss = compute_loss(X, y, theta)
        loss_history.append(loss)
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}")
    return theta, loss_history

# 4. 使用正规方程计算参数
theta_best = normal_equation(X_train, y_train)
print(f"Computed Parameters (Normal Equation): {theta_best.ravel()}")

# 5. 使用梯度下降优化参数
theta = np.random.randn(X_train.shape[1], 1)
learning_rate = 0.1
num_iterations = 1000

theta_optimal, loss_history = gradient_descent(
    X_train, y_train, theta, learning_rate, num_iterations
)
print(f"Optimized Parameters (Gradient Descent): {theta_optimal.ravel()}")

# 6. 模型评估
train_loss = compute_loss(X_train, y_train, theta_optimal)
test_loss = compute_loss(X_test, y_test, theta_optimal)
print(f"Training Loss: {train_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# 7. 可视化结果

# 数据和拟合结果
plt.scatter(X[:, 0], y)
X_plot = np.linspace(0, 2, 100)
X_plot_b = np.hstack((np.ones((100, 1)), X_plot.reshape(-1, 1)))
y_plot = X_plot_b @ theta_optimal
plt.plot(X_plot, y_plot, color='red')
plt.title('Linear Regression Fit (Gradient Descent)')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# 损失下降曲线
plt.plot(loss_history)
plt.title('Loss over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()


if __name__ == '__main__':
    pass