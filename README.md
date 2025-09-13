# 2022-Machine-Learning-Specialization

Andrew Ng机器学习专项课程学习笔记和练习代码。

## 课程大纲

- Supervised Machine Learning: Regression and Classification
- Advanced Learning Algorithms
- Unsupervised Learning: Recommenders, Reinforcement Learning

课程链接：https://www.coursera.org/learn/machine-learning/

## 项目结构

- `work/` - 课程官方代码
- `practise/` - 学习时的练习代码
  - `supervised_learning/` - 监督学习（线性回归、逻辑回归等）
  - `Advanced Learning Algorithms/` - 深度学习（CNN、RNN、Transformer等）
  - `py/` - Python工具脚本
- `images/` - 图片资源

## 环境配置（Conda）

```bash
# 创建环境
conda create -n machinelearning python=3.10 -y

# 激活环境
conda activate machinelearning

# 安装依赖
conda install numpy pandas matplotlib scikit-learn jupyter tensorflow -y
```

## 使用方法

```bash
# 激活环境
conda activate machinelearning

# 启动Jupyter
jupyter notebook
```