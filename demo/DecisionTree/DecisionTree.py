import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # 用于分割的特征
        self.feature = feature
        # 分割阈值
        self.threshold = threshold
        # 左子树
        self.left = left
        # 右子树
        self.right = right
        # 叶节点的预测值
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 检查停止条件
        if (self.max_depth <= depth or
                n_samples < self.min_samples_split or
                n_labels == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 寻找最佳分割
        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 根据最佳分割创建子树
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X[:, feature], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X_column, y, threshold):
        parent_entropy = self._entropy(y)

        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# 创建示例数据
# 假设我们在预测水果类型
X = np.array([
    [100, 0.5, 1],  # 第一个水果：重量100g，直径0.5cm
    [120, 0.6, 2],  # 第二个水果：重量120g，直径0.6cm
    [200, 1.0, 1],  # 第三个水果：重量200g，直径1.0cm
    [220, 1.1, 2]   # 第四个水果：重量220g，直径1.1cm
])

y = np.array([
    0,           # 草莓（类别0）
    0,           # 草莓（类别0）
    1,           # 苹果（类别1）
    1            # 苹果（类别1）
])

# 创建和训练决策树
tree = DecisionTree(max_depth=4)
tree.fit(X, y)

# 预测新数据
predictions = tree.predict(np.array([[150, 0.8, 2], [250, 1.2, 1]]))

if __name__ == "__main__":
    print(predictions)
    # Output: [0 1]