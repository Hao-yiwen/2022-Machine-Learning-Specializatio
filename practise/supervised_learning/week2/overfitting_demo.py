import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# 创建训练数据和测试数据
x_train = np.linspace(0, 4*np.pi, 20)
y_train = np.cos(x_train/2) + np.random.normal(0, 0.1, 20)  # 添加噪声

x_test = np.linspace(0, 4*np.pi, 100)
y_test = np.cos(x_test/2)  # 真实的余弦函数，无噪声

# 准备不同阶数的多项式
degrees = [1, 3, 5, 10, 15]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

train_errors = []
test_errors = []

for idx, degree in enumerate(degrees):
    # 创建多项式特征
    X_train = np.vander(x_train, degree + 1, increasing=True)[:, 1:]  # 去掉常数项
    X_test = np.vander(x_test, degree + 1, increasing=True)[:, 1:]
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # 计算误差
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_errors.append(train_mse)
    test_errors.append(test_mse)
    
    # 绘图
    ax = axes[idx]
    ax.scatter(x_train, y_train, c='red', s=50, alpha=0.5, label='训练数据')
    ax.plot(x_test, y_test, 'g--', alpha=0.5, label='真实函数')
    ax.plot(x_test, y_test_pred, 'b-', label=f'多项式拟合')
    ax.set_title(f'阶数 = {degree}\n训练MSE: {train_mse:.4f}, 测试MSE: {test_mse:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 2)

# 绘制误差对比图
ax = axes[5]
x_pos = np.arange(len(degrees))
width = 0.35

bars1 = ax.bar(x_pos - width/2, train_errors, width, label='训练误差', color='blue', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, test_errors, width, label='测试误差', color='red', alpha=0.7)

ax.set_xlabel('多项式阶数')
ax.set_ylabel('均方误差 (MSE)')
ax.set_title('训练误差 vs 测试误差')
ax.set_xticks(x_pos)
ax.set_xticklabels(degrees)
ax.legend()
ax.grid(True, alpha=0.3)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('/home/haoyiwen/Documents/ai/2022-Machine-Learning-Specializatio/practise/supervised_learning/week2/overfitting_comparison.png', dpi=150)
plt.show()

# 打印详细的过拟合分析
print("="*60)
print("过拟合分析报告")
print("="*60)
for i, degree in enumerate(degrees):
    print(f"\n多项式阶数: {degree}")
    print(f"  训练误差: {train_errors[i]:.6f}")
    print(f"  测试误差: {test_errors[i]:.6f}")
    print(f"  过拟合程度: {test_errors[i]/train_errors[i]:.2f}x")
    
    if degree <= 3:
        print("  状态: ✓ 欠拟合 - 模型太简单")
    elif degree <= 5:
        print("  状态: ✓ 良好拟合 - 平衡的复杂度")
    else:
        print("  状态: ✗ 过拟合 - 模型过于复杂")

print("\n" + "="*60)
print("关键观察:")
print("1. 低阶多项式(1-3阶): 训练和测试误差都较高 → 欠拟合")
print("2. 中阶多项式(5阶): 训练和测试误差都较低且接近 → 良好拟合")
print("3. 高阶多项式(10-15阶): 训练误差很低，测试误差激增 → 过拟合")
print("="*60)