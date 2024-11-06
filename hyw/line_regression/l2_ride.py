import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# 1. 生成示例数据
def generate_sample_data(n_samples=100, n_features=10):
    np.random.seed(42)
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    # 真实权重
    true_weights = np.array([1.0, 0.8, -0.6, 0.4, -0.2] + [0.1] * 5)
    # 生成目标值（加入噪声）
    y = np.dot(X, true_weights) + np.random.randn(n_samples) * 0.1
    return X, y, true_weights


# 2. 对比普通线性回归和Ridge回归
def compare_models(X_train, X_test, y_train, y_test, alphas):
    results = []

    # 对每个alpha值训练Ridge模型
    for alpha in alphas:
        # Ridge模型
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)

        # 计算训练和测试得分
        train_pred = ridge.predict(X_train)
        test_pred = ridge.predict(X_test)

        result = {
            'alpha': alpha,
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'r2_score': r2_score(y_test, test_pred),
            'coefficients': ridge.coef_
        }
        results.append(result)

    return pd.DataFrame(results)


# 3. 可视化函数
def plot_results(results, true_weights):
    # 1. MSE随alpha变化的曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.semilogx(results['alpha'], results['train_mse'], 'b-', label='Train MSE')
    plt.semilogx(results['alpha'], results['test_mse'], 'r-', label='Test MSE')
    plt.xlabel('alpha (log scale)')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs Alpha')
    plt.legend()

    # 2. 系数变化图
    plt.subplot(1, 2, 2)
    for i in range(len(true_weights)):
        plt.semilogx(results['alpha'],
                     [coef[i] for coef in results['coefficients']],
                     label=f'Feature {i}')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('alpha (log scale)')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficient Paths')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


# 主程序
def main():
    # 1. 生成数据
    X, y, true_weights = generate_sample_data(n_samples=100, n_features=10)

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 设置不同的alpha值
    alphas = np.logspace(-3, 3, 100)

    # 5. 训练和评估模型
    results = compare_models(
        X_train_scaled, X_test_scaled, y_train, y_test, alphas
    )

    # 6. 找出最佳alpha
    best_alpha = results.loc[results['test_mse'].idxmin(), 'alpha']
    print(f"Best alpha: {best_alpha}")

    # 7. 使用最佳alpha训练最终模型
    best_model = Ridge(alpha=best_alpha)
    best_model.fit(X_train_scaled, y_train)

    # 8. 打印最终模型的评估结果
    print("\nFinal Model Performance:")
    print(f"R2 Score: {r2_score(y_test, best_model.predict(X_test_scaled)):.4f}")
    print(f"Test MSE: {mean_squared_error(y_test, best_model.predict(X_test_scaled)):.4f}")

    # 9. 对比真实权重和预测权重
    print("\nWeight Comparison:")
    for i, (true_w, pred_w) in enumerate(zip(true_weights, best_model.coef_)):
        print(f"Feature {i}: True = {true_w:.4f}, Predicted = {pred_w:.4f}")

    # 10. 可视化结果
    plot_results(results, true_weights)


if __name__ == "__main__":
    main()