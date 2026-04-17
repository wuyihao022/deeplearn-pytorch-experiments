# Experiment 02 / 实验 02: Linear Regression / 线性回归

## Goal / 实验目标

Use PyTorch to fit a straight line from synthetic data.  
用 `PyTorch` 拟合一条直线，理解最基础的监督学习训练流程。

You will learn these ideas:  
你会学到这几个核心概念：

- model / 模型
- loss function / 损失函数
- optimizer / 优化器
- epoch / 训练轮次
- parameter update / 参数更新

## Task / 实验任务

We create synthetic data with a hidden rule:  
我们先造一批带规律的数据：

```text
y = 3x + 2 + noise
```

Then we train a linear model to recover that rule.  
然后训练一个线性模型，让它自己学出这条规律。

## Files / 文件说明

- `main.py`: train the linear regression model  
  `main.py`：训练线性回归模型
- `linear_regression_fit.png`: output figure after training  
  `linear_regression_fit.png`：训练完成后保存的拟合图

## How to run / 如何运行

```bash
python main.py
```

## What the code is doing / 代码在做什么

1. Generate input data `x`  
   生成输入数据 `x`
2. Generate target data `y`  
   按 `y = 3x + 2 + noise` 生成目标数据 `y`
3. Build a model with `nn.Linear(1, 1)`  
   用 `nn.Linear(1, 1)` 建一个最简单的线性模型
4. Use `MSELoss` to measure prediction error  
   用 `MSELoss` 衡量预测误差
5. Use `SGD` to update parameters  
   用 `SGD` 更新模型参数
6. Plot the learned line against the real data  
   把学出来的直线和原始数据画出来

## Key ideas / 核心概念

### 1. Model / 模型

The model is:

```text
y = wx + b
```

模型本质上就是：

```text
y = wx + b
```

Here:

- `w` is weight / `w` 是权重
- `b` is bias / `b` 是偏置

### 2. Loss Function / 损失函数

We use `MSELoss`, which measures how far predictions are from targets.  
这里用的是 `MSELoss`，它衡量“预测值”和“真实值”差得有多远。

### 3. Optimizer / 优化器

We use `SGD`, which updates `w` and `b` using gradients.  
这里用的是 `SGD`，它会根据梯度去更新 `w` 和 `b`。

### 4. One training step / 一次训练步骤

Each training step does this:

1. forward pass / 前向计算
2. compute loss / 计算损失
3. `loss.backward()` / 反向传播算梯度
4. `optimizer.step()` / 更新参数
5. `optimizer.zero_grad()` / 清空旧梯度

## What to watch / 你要重点观察什么

- loss becomes smaller during training  
  训练过程中 `loss` 会越来越小
- learned `weight` gets close to `3`  
  学到的 `weight` 会接近 `3`
- learned `bias` gets close to `2`  
  学到的 `bias` 会接近 `2`

## Expected takeaway / 学完后你应该会什么

By the end of this experiment, you should be able to explain:

- why this is called linear regression  
  为什么这叫线性回归
- what the model is learning  
  模型到底在学习什么
- what `loss.backward()` does  
  `loss.backward()` 在干什么
- how `optimizer.step()` changes parameters  
  `optimizer.step()` 为什么能更新参数

## Suggested next step / 下一步建议

After running this experiment, try changing:

- learning rate / 学习率
- number of epochs / 训练轮数
- noise size / 噪声大小

这样你会更容易真正理解训练过程。
