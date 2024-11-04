import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 数据准备
X, y = make_classification(
    n_samples=5000,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
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


# 转换标签为独热编码
def one_hot_encode(y, num_classes):
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot


num_classes = np.unique(y_train).shape[0]
y_train_encoded = one_hot_encode(y_train, num_classes)
y_test_encoded = one_hot_encode(y_test, num_classes)

# 添加偏置项
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))


# 3. 定义模型和函数
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_loss(X, y, theta):
    m = y.shape[0]
    logits = X @ theta
    y_pred = softmax(logits)
    loss = - (1 / m) * np.sum(y * np.log(y_pred + 1e-5))
    return loss


def compute_gradient(X, y, theta):
    m = y.shape[0]
    logits = X @ theta
    y_pred = softmax(logits)
    gradient = (1 / m) * X.T @ (y_pred - y)
    return gradient


def predict(X, theta):
    logits = X @ theta
    y_pred = softmax(logits)
    return np.argmax(y_pred, axis=1)


# 4. 训练模型
num_features = X_train.shape[1]
num_classes = y_train_encoded.shape[1]

theta = np.zeros((num_features, num_classes))
learning_rate = 0.1
num_iterations = 5000
loss_history = []

for i in range(num_iterations):
    gradient = compute_gradient(X_train, y_train_encoded, theta)
    theta -= learning_rate * gradient
    loss = compute_loss(X_train, y_train_encoded, theta)
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

# 6. 绘制损失曲线
plt.plot(loss_history)
plt.title('Loss over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()


if __name__ == '__main__':
    pass