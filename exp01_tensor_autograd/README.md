# Experiment 01 / 实验 01: Tensor and Autograd / 张量与自动求导

## Goal / 实验目标

Learn the PyTorch basics that everything else builds on.  
学习 `PyTorch` 最基础、最核心的内容，后面的实验都会建立在这些概念上。

- create tensors  
  创建张量 `tensor`
- inspect shapes and dtypes  
  查看张量的形状 `shape` 和数据类型 `dtype`
- do basic math  
  做基本数学运算
- perform matrix multiplication  
  做矩阵乘法
- understand `requires_grad`  
  理解 `requires_grad` 的作用
- run backpropagation with `backward()`  
  用 `backward()` 做反向传播

## Files / 文件说明

- `main.py`: runnable walkthrough for this experiment  
  `main.py`：这个实验的可运行示例脚本

## How to run / 如何运行

```bash
python main.py
```

## What to focus on / 你要重点看什么

When you run the script, pay attention to these points.  
运行脚本时，重点观察下面这些内容：

1. What a tensor looks like compared with a Python list  
   `tensor` 和普通 Python 列表看起来有什么不同
2. How tensor shapes change after `reshape`  
   `reshape` 之后张量形状是怎么变化的
3. Why matrix multiplication requires compatible shapes  
   为什么矩阵乘法要求维度必须匹配
4. What `requires_grad=True` means  
   `requires_grad=True` 到底表示什么
5. Why `x.grad` appears only after `backward()`  
   为什么 `x.grad` 要在 `backward()` 之后才会有值

## Key ideas / 核心概念

### 1. Tensor / 张量

A tensor is the basic data object in PyTorch.  
`tensor` 是 `PyTorch` 里最基本的数据对象。

You can treat it as a more powerful version of a list or matrix.  
你可以先把它理解成“更强的数组”或者“更通用的矩阵”。

Examples:  
例如：

- `[1, 2, 3]` can become a 1D tensor  
  `[1, 2, 3]` 可以变成一个一维张量
- a table of numbers can become a 2D tensor  
  一个数字表格可以变成二维张量
- an image can also be represented as a tensor  
  一张图片本质上也可以表示成张量

### 2. Shape / 形状

The `shape` tells you the size of each dimension.  
`shape` 表示这个张量每个维度有多大。

For example:  
比如：

- `torch.Size([3])` means a 1D tensor with 3 elements  
  `torch.Size([3])` 表示一个有 3 个元素的一维张量
- `torch.Size([2, 3])` means 2 rows and 3 columns  
  `torch.Size([2, 3])` 表示 2 行 3 列

### 3. Autograd / 自动求导

Autograd is the system that helps PyTorch compute gradients automatically.  
自动求导是 `PyTorch` 帮你自动计算梯度的机制。

This is very important in deep learning because training a model needs gradients.  
这在深度学习里非常重要，因为训练模型时必须计算梯度。

### 4. Gradient / 梯度

A gradient tells you how much the output changes when the input changes a little.  
梯度可以理解成：输入变化一点点，输出会跟着变化多少。

In training, gradients are used to update model parameters.  
在训练过程中，梯度用来更新模型参数。

## Practice tasks / 练习任务

After you finish the provided script, try these by yourself.  
把给你的脚本跑完以后，自己再做下面这些小练习：

1. Change the vector values and rerun the script  
   改一下向量里的数字，再重新运行
2. Create a `3 x 3` matrix and print its shape  
   创建一个 `3 x 3` 的矩阵，并打印它的形状
3. Replace the function `y = x^2 + 2x + 1` with `y = 3x^2 + x`  
   把函数 `y = x^2 + 2x + 1` 改成 `y = 3x^2 + x`
4. Verify the gradient by hand and compare it with PyTorch  
   自己手算梯度，再和 `PyTorch` 的结果对比
5. Add one more matrix multiplication example  
   再补一个矩阵乘法例子

## Expected takeaway / 学完后你应该会什么

By the end of this experiment, you should be able to explain:  
做完这个实验后，你应该能用自己的话解释：

- what a tensor is  
  什么是 `tensor`
- what a shape is  
  什么是 `shape`
- what a gradient is  
  什么是梯度
- what autograd is doing for you  
  自动求导到底帮你做了什么

## Suggested next step / 下一步建议

After reading the README, run the script and compare the output with the explanations here.  
看完这个 README 之后，直接运行脚本，把终端输出和这里的解释一条条对照起来看。

If you want, the next thing I can do is make `main.py` bilingual too, with Chinese comments and Chinese output.  
如果你愿意，我下一步还可以把 `main.py` 也改成双语版，给你补中文注释和中文输出。
