# Experiment 06: Overfitting and Regularization / 实验 06：过拟合与正则化

## Goal / 实验目标

This experiment shows what overfitting looks like in real training curves.

本实验的目标是让你从真实训练曲线中看懂“过拟合”到底长什么样。

We intentionally train on a small MNIST training subset, so a large MLP can
memorize the training samples more easily. Then we compare several common ways
to improve generalization.

这里会故意只使用一小部分 `MNIST` 训练样本。训练集变小以后，一个比较大的
MLP 很容易把训练样本“背下来”，于是训练集效果很好，但验证集/测试集效果不一定好。
这样就能更明显地观察过拟合现象。

This experiment compares:

本实验会对比：

- no regularization / 不加正则化
- Dropout / 随机失活
- weight decay / 权重衰减
- smaller model / 减小模型规模
- early stopping / 提前停止

## Files / 文件说明

```text
exp06_overfitting_regularization/
  main.py       # run all regularization comparisons / 运行所有正则化对比实验
  outputs/
    regularization_comparison.png  # training curves / 训练曲线图
    summary.csv                    # final metrics / 最终结果表
  data/         # MNIST download cache / MNIST 数据缓存
```

## How to Run / 如何运行

Run the full experiment:

运行完整实验：

```bash
python main.py
```

Useful faster test run:

如果只是想快速确认代码能跑，可以用更小的配置：

```bash
python main.py --epochs 3 --train-size 500 --val-size 1000 --test-size 1000
```

Common settings:

常用参数：

```bash
python main.py --epochs 15 --batch-size 128 --learning-rate 0.001
```

Arguments:

参数含义：

- `--epochs`: number of training epochs / 训练轮数
- `--batch-size`: batch size / 每个 batch 的样本数
- `--learning-rate`: learning rate / 学习率
- `--train-size`: number of training samples / 训练集样本数
- `--val-size`: number of validation samples / 验证集样本数
- `--test-size`: number of test samples / 测试集样本数
- `--seed`: random seed / 随机种子

## What to Watch / 重点观察什么

The script saves:

脚本运行结束后会保存：

- `outputs/regularization_comparison.png`
- `outputs/summary.csv`

In the curve image, focus on the baseline model first.

看曲线时，先观察 `baseline_big_model`，也就是不加正则化的大模型。

Signs of overfitting:

过拟合的典型现象：

- training loss keeps going down / 训练损失继续下降
- validation loss stops decreasing or starts increasing / 验证损失不再下降，甚至开始上升
- training accuracy is much higher than validation accuracy / 训练准确率明显高于验证准确率

In plain words:

通俗地说：

```text
训练集表现越来越好，但验证集表现没有同步变好，
说明模型可能不是学到了通用规律，而是在记忆训练集。
```

## Regularization Ideas / 正则化方法理解

`Dropout`

Randomly disables some hidden units during training. This makes the network less
dependent on one exact path through the model.

训练时随机关闭一部分神经元，让模型不能过度依赖某几个固定神经元组合。
它的作用是减少神经元之间的“互相依赖”，让模型更不容易死记硬背。

`weight decay`

Adds a penalty for large weights. This discourages overly sharp or complicated
solutions.

对过大的权重进行惩罚，让模型参数不要变得太极端。权重太大时，模型往往会对训练数据中的
细节和噪声过于敏感，`weight decay` 可以让模型更平滑一些。

`small_model`

Uses fewer hidden units. A smaller model has less capacity, so it is harder for
it to memorize a small training set.

减少隐藏层规模，降低模型容量。模型越大，记忆能力越强；模型变小后，没那么容易把小训练集
完全背下来。

`early_stopping`

Stops training when validation loss stops improving. This prevents the model
from continuing to memorize the training set after validation performance has
peaked.

当验证集损失一段时间不再变好时，提前停止训练。它的核心思想是：

```text
验证集效果最好的那一刻，通常比训练到最后更值得保留。
```

## Output Interpretation / 输出结果怎么看

`regularization_comparison.png`

This figure compares training loss, validation loss, and validation accuracy
across different methods.

这张图用来比较不同方法的训练损失、验证损失和验证准确率。重点看：

- 哪个方法的验证损失更低
- 哪个方法的验证准确率更高
- 哪个方法的训练集和验证集差距更小

`summary.csv`

This table records the final metrics for each method.

这个表格记录每种方法的最终指标，包括：

- `best_epoch`: validation loss 最好的 epoch
- `final_train_acc`: 最后一轮训练准确率
- `final_val_acc`: 最后一轮验证准确率
- `train_val_acc_gap`: 训练准确率和验证准确率的差距
- `test_loss`: 测试集损失
- `test_acc`: 测试集准确率

If `train_val_acc_gap` is very large, the model may be overfitting.

如果 `train_val_acc_gap` 很大，说明模型在训练集上表现明显好于验证集，可能存在过拟合。

## Expected Takeaway / 学完后应该掌握什么

After this experiment, you should be able to explain:

完成本实验后，你应该能解释：

- what overfitting looks like in train and validation curves / 过拟合在训练曲线和验证曲线中是什么样子
- why training accuracy alone is not enough / 为什么不能只看训练准确率
- how Dropout, weight decay, model size, and early stopping reduce overfitting / 不同正则化方法分别如何缓解过拟合
- why validation data is used to choose training decisions / 为什么要用验证集辅助训练决策

The most important idea:

最重要的一句话：

```text
模型不是训练集分数越高越好，而是要在没见过的数据上也表现稳定。
```
