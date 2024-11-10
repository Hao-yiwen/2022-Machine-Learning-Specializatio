import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# 1. 数据准备
X, y = make_classification(
    n_samples=10000,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# 2. 数据预处理
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))


# 3. 定义模型和函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_loss(X, y, theta):
    m = y.size
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    loss = - (1 / m) * np.sum(
        y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)
    )
    return loss


def compute_gradient(X, y, theta):
    m = y.size
    h = sigmoid(X @ theta)
    gradient = (1 / m) * X.T @ (h - y)
    return gradient


def predict(X, theta, threshold=0.5):
    probabilities = sigmoid(X @ theta)
    return (probabilities >= threshold).astype(int)


def plot_decision_boundary(X, y, theta):
    x_values = [np.min(X[:, 1] - 0.5), np.max(X[:, 1] + 0.5)]
    y_values = - (theta[0] + np.dot(theta[1], x_values)) / theta[2]

    plt.scatter(X[:, 1], X[:, 2], c=y, cmap='bwr', alpha=0.7)
    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


# 4. 训练模型
theta = np.zeros(X_train.shape[1])
learning_rate = 0.1
num_iterations = 1000
loss_history = []

for i in range(num_iterations):
    gradient = compute_gradient(X_train, y_train, theta)
    print(f"theta: {theta}")
    theta -= learning_rate * gradient
    loss = compute_loss(X_train, y_train, theta)
    loss_history.append(loss)

    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss:.4f}")

# 5. 模型评估
y_train_pred = predict(X_train, theta)
train_accuracy = np.mean(y_train_pred == y_train) * 100
print(f"Training Accuracy: {train_accuracy:.2f}%")

y_test_pred = predict(X_test, theta)
test_accuracy = np.mean(y_test_pred == y_test) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# 6. 可视化结果
plt.plot(loss_history)
plt.title('Loss over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

plot_decision_boundary(X_train, y_train, theta)

def main():
    pass

if __name__ == "__main__":
    main()