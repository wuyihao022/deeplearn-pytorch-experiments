from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


def print_section(title: str) -> None:
    # 打印一个分节标题，方便看终端输出
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def generate_data() -> tuple[torch.Tensor, torch.Tensor]:
    # 固定随机种子，这样你每次运行时生成的数据都一样generate_data
    torch.manual_seed(42)

    # 生成 100 个 x 值，范围从 -5 到 5
    # 原始 shape 是 [100]，unsqueeze(1) 之后变成 [100, 1]
    # 这里可以理解成：100 条样本，每条样本只有 1 个特征
    x = torch.linspace(-5, 5, 100).unsqueeze(1)

    # 生成噪声，shape 也是 [100, 1]
    # 每个 x 对应一个噪声值
    noise = torch.randn(100, 1) * 0.8

    # 按照公式 y = 3x + 2 + noise 生成目标值
    # 所以这里本质上是：每一个 x，都对应一个 y
    y = 3 * x + 2 + noise

    # 返回输入 x 和目标 y
    return x, y


def train_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 200,
    learning_rate: float = 0.05,
) -> list[float]:
    # 均方误差损失：预测值和真实值差得越大，loss 越大
    criterion = nn.MSELoss()

    # SGD 优化器：根据梯度更新模型参数
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 用来保存每一轮的 loss，后面画图会用到
    loss_history: list[float] = []

    print_section("1. Training / 训练开始")

    for epoch in range(1, epochs + 1):
        # 前向计算：把 x 输入模型，得到预测值
        predictions = model(x)

        # 计算损失：看预测值和真实 y 差多少
        loss = criterion(predictions, y)

        # 清空上一次留下的梯度
        # PyTorch 默认会累加梯度，所以每轮都要先清零
        optimizer.zero_grad()

        # 反向传播：自动计算 loss 对模型参数的梯度
        loss.backward()

        # 用优化器根据梯度更新参数
        optimizer.step()

        # 记录当前这轮的 loss 数值
        loss_history.append(loss.item())

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            # 取出当前学到的 weight 和 bias，方便观察训练过程
            weight = model.weight.item()
            bias = model.bias.item()
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {loss.item():.4f} | "
                f"Weight: {weight:.4f} | "
                f"Bias: {bias:.4f}"
            )

    return loss_history


def save_plot(
    x: torch.Tensor,
    y: torch.Tensor,
    predictions: torch.Tensor,
    loss_history: list[float],
    output_path: Path,
) -> None:
    # 创建一张图，里面放两个子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左边：画真实数据散点图
    axes[0].scatter(x.numpy(), y.numpy(), label="Real data", alpha=0.7)

    # 左边：再画模型学到的拟合直线
    axes[0].plot(
        x.numpy(),
        predictions.numpy(),
        color="red",
        linewidth=2,
        label="Predicted line",
    )
    # 设置左图标题和坐标轴标签
    axes[0].set_title("Linear Regression Fit")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()

    # 右边：画训练过程中 loss 的变化曲线
    axes[1].plot(loss_history, color="green")
    axes[1].set_title("Loss Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")

    # 自动调整布局，避免文字挤在一起
    fig.tight_layout()

    # 保存图片到文件
    fig.savefig(output_path, dpi=150)

    # 关闭图像，释放内存
    plt.close(fig)


def main() -> None:
    # 打印当前 PyTorch 版本
    print("PyTorch version:", torch.__version__)

    print_section("0. Generate Data / 生成数据")

    # 生成数据集：x 是输入，y 是真实答案
    x, y = generate_data()

    # 打印 x 和 y 的形状
    print("x shape:", x.shape)
    print("y shape:", y.shape)

    # 打印前 5 个输入值
    print("First 5 x values / 前 5 个 x:")
    print(x[:5].squeeze())

    # 打印前 5 个目标值
    print("First 5 y values / 前 5 个 y:")
    print(y[:5].squeeze())

    print_section("1. Build Model / 创建模型")

    # 创建一个最简单的线性层：1 个输入，1 个输出
    # 它对应的公式就是 y = wx + b
    model = nn.Linear(1, 1)

    # 打印模型结构和初始化参数
    print(model)
    print("Initial weight / 初始权重:", model.weight.item())
    print("Initial bias / 初始偏置:", model.bias.item())

    # 开始训练，并拿到每一轮的 loss 记录
    loss_history = train_model(model, x, y)

    print_section("2. Final Parameters / 最终参数")

    # 训练结束后，读取模型学到的最终参数
    learned_weight = model.weight.item()
    learned_bias = model.bias.item()
    print(f"Learned weight / 学到的权重: {learned_weight:.4f}")
    print(f"Learned bias / 学到的偏置: {learned_bias:.4f}")
    print("Target weight / 真实权重: 3.0000")
    print("Target bias / 真实偏置: 2.0000")

    # 推理阶段不需要计算梯度，所以用 no_grad 包起来
    with torch.no_grad():
        # 用训练好的模型重新对所有 x 做一次预测
        predictions = model(x)

    # 生成输出图片路径
    output_path = Path(__file__).with_name("linear_regression_fit.png")

    # 保存拟合图和 loss 曲线图
    save_plot(x, y, predictions, loss_history, output_path)

    print_section("3. Saved Output / 已保存结果")
    print("Figure saved to / 图片已保存到:")
    print(output_path)

    print_section("4. Summary / 小结")
    print("This experiment shows the full training loop of a simple model.")
    print("这个实验展示了一个最基础模型的完整训练流程。")
    print("Next step: explain forward, loss, backward, and step in your own words.")
    print("下一步：试着用你自己的话解释 forward、loss、backward、step。")


if __name__ == "__main__":
    # 只有直接运行这个文件时，才执行 main()
    main()
