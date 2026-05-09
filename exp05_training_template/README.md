# Experiment 05: Standard Training Template / 实验 05：训练流程标准化

## Goal / 实验目标

This experiment refactors the MNIST MLP experiment into a reusable training
template.

本实验把前面的 MNIST MLP 实验重构成一套可以重复使用的训练模板。

The point is not to build a new model. The point is to learn a clean training
workflow that can be reused in later experiments.

重点不是换一个新模型，而是学会把训练流程拆清楚。以后做 CNN、正则化、
优化器对比、学习率实验时，都可以复用这套结构。

## File Structure / 文件结构

```text
exp05_training_template/
  dataset.py       # build train / validation / test DataLoader objects
  model.py         # define the MLP model
  train.py         # train_one_epoch, validate, fit
  inference.py     # predict and save prediction examples
  utils.py         # checkpoint and plotting helpers
  main.py          # connect everything together
  checkpoints/     # generated after training
  outputs/         # generated figures
  data/            # MNIST download cache
```

说明：

- `dataset.py`：负责数据集和 `DataLoader`
- `model.py`：负责定义神经网络模型
- `train.py`：负责训练和验证
- `inference.py`：负责预测和保存预测示例
- `utils.py`：负责保存 checkpoint、画图等工具函数
- `main.py`：负责把所有模块串起来

## Key Functions / 关键函数

`train_one_epoch`

Runs one full pass over the training set and updates model parameters.

完整跑一遍训练集，并且更新模型参数。真正的“学习”发生在这里：

```python
loss.backward()
optimizer.step()
```

`validate`

Evaluates the model without updating parameters.

在验证集或测试集上评估模型，不更新参数。这里会使用：

```python
model.eval()
with torch.no_grad():
```

`predict`

Turns model outputs into predicted classes.

把模型输出的 logits 转成最终预测类别。

`save_checkpoint`

Saves the best model state, optimizer state, epoch, validation accuracy, and
training history.

保存当前最好的模型参数、优化器状态、训练轮数、验证集准确率和训练历史。

## How to Run / 如何运行

```bash
python main.py
```

默认训练 5 个 epoch。

You can also change common settings:

也可以手动修改常见训练参数：

```bash
python main.py --epochs 5 --batch-size 128 --learning-rate 0.001
```

## What to Watch / 重点观察什么

The script records four curves:

脚本会记录四条曲线：

- train loss
- validation loss
- train accuracy
- validation accuracy

After training, it saves:

训练完成后会保存：

- `checkpoints/best_mlp.pt`
- `outputs/training_history.png`
- `outputs/prediction_examples.png`

其中：

- `best_mlp.pt`：验证集效果最好的模型
- `training_history.png`：训练 loss / 验证 loss / 训练准确率 / 验证准确率曲线
- `prediction_examples.png`：模型在测试集上的预测示例

## Training Flow / 标准训练流程

一次完整训练通常包括：

```text
1. 准备数据
2. 创建模型
3. 选择损失函数
4. 选择优化器
5. 训练一个 epoch
6. 在验证集上评估
7. 如果验证效果更好，就保存 checkpoint
8. 训练结束后，在测试集上评估最佳模型
9. 保存曲线和预测结果
```

对应代码结构是：

```text
dataset.py   -> 准备数据
model.py     -> 创建模型
train.py     -> 训练和验证
utils.py     -> 保存 checkpoint 和曲线
inference.py -> 预测
main.py      -> 总入口
```

## Expected Takeaway / 学完后应该掌握什么

After this experiment, you should understand the standard PyTorch training
workflow:

```text
load data -> build model -> choose loss -> choose optimizer
-> train -> validate -> save best checkpoint -> test -> predict
```

This structure is the base template for later experiments.

学完这个实验后，你应该能说清楚：

- 为什么训练集、验证集、测试集要分开
- `train_one_epoch` 里模型是怎么学习的
- `validate` 为什么不能更新参数
- 为什么要保存验证集效果最好的 checkpoint
- 后续实验如何复用这套训练模板
