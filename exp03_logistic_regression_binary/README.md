# Experiment 03 / 实验 03: Logistic Regression Binary Classification / 逻辑回归二分类

## Goal / 实验目标

Use PyTorch to train a simple binary classifier.

用 `PyTorch` 训练一个最简单的二分类模型，理解分类问题和回归问题的区别。

You will learn these ideas:

- binary classification / 二分类
- logit / 原始分类分数
- sigmoid / 把分数变成概率
- threshold / 分类阈值
- accuracy / 准确率
- `BCEWithLogitsLoss`

## Task / 实验任务

We create two groups of 2D points:

我们生成人造二维点集：

```text
class 0: around (-2, -2)
class 1: around ( 2,  2)
```

Then we train a logistic regression model to separate the two groups.

然后训练一个逻辑回归模型，让它学会把两类点分开。

## Files / 文件说明

- `main.py`: train the binary classification model
- `classification_boundary.png`: output figure after training

## How to run / 如何运行

```bash
python main.py
```

## What the code is doing / 代码在做什么

1. Generate two groups of 2D points
2. Give class labels: `0` and `1`
3. Build a model with `nn.Linear(2, 1)`
4. Use `BCEWithLogitsLoss` to measure classification error
5. Use `SGD` to update model parameters
6. Use `sigmoid` to turn logits into probabilities
7. Use `0.5` as the threshold to get final classes
8. Plot the decision boundary

## Key ideas / 核心概念

### 1. Linear regression vs logistic regression / 线性回归和逻辑回归

Experiment 02 predicts a continuous number:

实验 02 预测的是连续数值：

```text
y = wx + b
```

Experiment 03 predicts a category:

实验 03 预测的是类别：

```text
class 0 or class 1
```

The model still starts with a linear formula:

模型一开始仍然是线性公式：

```text
logit = w1*x1 + w2*x2 + b
```

But now the output is used for classification.

### 2. Logit / 原始分数

The model output is called a `logit`.

模型直接输出的值叫 `logit`，它还不是概率。

Example:

```text
logit = 2.3
logit = -1.5
```

A larger logit means the model thinks the sample is more likely to be class `1`.

`logit` 越大，模型越倾向于认为这个样本是类别 `1`。

### 3. Sigmoid / 概率

`sigmoid` turns any logit into a number between `0` and `1`.

`sigmoid` 会把任意分数压到 `0` 到 `1` 之间：

```text
probability = sigmoid(logit)
```

Then we can read it as probability:

```text
0.93 means likely class 1
0.08 means likely class 0
```

### 4. Threshold / 阈值

After getting probability, we use `0.5` as the threshold:

拿到概率之后，用 `0.5` 做分界线：

```text
probability >= 0.5 -> class 1
probability <  0.5 -> class 0
```

### 5. Why `BCEWithLogitsLoss`? / 为什么用 BCEWithLogitsLoss？

For binary classification, a common beginner version is:

二分类里，一个容易想到的写法是：

```python
probability = torch.sigmoid(logit)
loss = BCELoss(probability, label)
```

But PyTorch usually recommends:

但实际更推荐：

```python
loss = BCEWithLogitsLoss(logit, label)
```

Because `BCEWithLogitsLoss` combines sigmoid and binary cross entropy in a more stable way.

因为 `BCEWithLogitsLoss` 把 `sigmoid` 和二分类交叉熵合在一起，数值上更稳定。

## What to watch / 重点观察

- loss should become smaller during training
- accuracy should become high
- the decision boundary should separate the two point groups
- `nn.Linear(2, 1)` means 2 input features and 1 output logit

## Expected takeaway / 学完后你应该会什么

After this experiment, you should be able to explain:

- why classification is different from regression
- what a logit is
- why sigmoid can be treated as probability
- how `0.5` becomes the decision threshold
- why `BCEWithLogitsLoss` is preferred over manual `sigmoid + BCELoss`

## Suggested next step / 下一步建议

Try changing:

- the point centers
- the noise size
- the learning rate
- the number of epochs

观察这些变化会怎样影响 loss、准确率和分类边界。
