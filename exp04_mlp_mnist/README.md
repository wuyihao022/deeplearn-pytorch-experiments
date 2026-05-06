# Experiment 04 / 实验 04: MLP MNIST Handwritten Digit Recognition / 多层感知机识别手写数字

## Goal / 实验目标

Train a neural network to recognize handwritten digits from `0` to `9`.

训练一个神经网络识别 `0` 到 `9` 的手写数字。

This is the first more complete deep learning experiment in this folder.

这是第一个比较完整的深度学习实验。

## Dataset / 数据集

We use `MNIST`.

`MNIST` 里面每张图片是：

```text
28 x 28 grayscale image
```

每张图片的标签是：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, or 9
```

## Model / 模型

The model is an MLP, also called a fully connected neural network.

模型是多层感知机，也可以叫全连接神经网络。

Structure:

```text
image: [1, 28, 28]
Flatten -> 784 numbers
Linear(784, 128)
ReLU
Linear(128, 64)
ReLU
Linear(64, 10)
```

The final `10` outputs mean:

最后输出 `10` 个分数，分别对应：

```text
digit 0, digit 1, digit 2, ..., digit 9
```

## Files / 文件说明

- `main.py`: train and evaluate the MLP
- `mnist_predictions.png`: prediction examples after training
- `training_curves.png`: loss and accuracy curves
- `data/`: MNIST dataset files downloaded by torchvision

## How to run / 如何运行

```bash
python main.py
```

The first run will download MNIST, so it may take longer.

第一次运行会下载 `MNIST` 数据集，可能会慢一点。

## Key ideas / 核心概念

### 1. Dataset / 数据集

`datasets.MNIST(...)` loads the MNIST dataset.

It gives us many pairs:

```text
image, label
```

For example:

```text
image = handwritten digit picture
label = 7
```

### 2. DataLoader / 数据加载器

`DataLoader` does not give the model one image at a time.

`DataLoader` 会按 batch 给数据。

Example:

```text
images shape = [128, 1, 28, 28]
labels shape = [128]
```

Meaning:

```text
128 images in one batch
each image has 1 channel
each image size is 28 x 28
```

### 3. Flatten / 展平

The model cannot feed a `[1, 28, 28]` image directly into `Linear(784, 128)`.

So we use:

```python
nn.Flatten()
```

It changes one image from:

```text
[1, 28, 28]
```

to:

```text
[784]
```

because:

```text
28 * 28 = 784
```

### 4. CrossEntropyLoss / 多分类损失函数

For experiment 03 binary classification, we used:

```python
nn.BCEWithLogitsLoss()
```

For experiment 04 ten-class classification, we use:

```python
nn.CrossEntropyLoss()
```

Important:

`CrossEntropyLoss` expects raw logits, not probabilities.

So the model does not need `softmax` inside the last layer.

### 5. train and eval modes / 训练模式和评估模式

During training:

```python
model.train()
```

During testing:

```python
model.eval()
```

When testing, we also use:

```python
with torch.no_grad():
```

because testing only checks performance and does not update model parameters.

## What to watch / 重点观察

- train loss should go down
- test accuracy should go up
- final output shape should be `[batch_size, 10]`
- `argmax(dim=1)` picks the class with the largest score

## Expected takeaway / 学完后你应该会什么

After this experiment, you should be able to explain:

- why a `28 x 28` image becomes `784` numbers
- why the final layer outputs `10` numbers
- what `DataLoader` gives to the model each step
- why multi-class classification uses `CrossEntropyLoss`
- what `model.train()` and `model.eval()` do

## Suggested next step / 下一步建议

Try changing:

- hidden layer size, such as `128` to `256`
- batch size
- learning rate
- number of epochs

Then compare the final accuracy.
