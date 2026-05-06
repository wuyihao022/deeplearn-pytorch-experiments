from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


def print_section(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def generate_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)

    samples_per_class = 100

    # Class 0: points around the lower-left area.
    class0 = torch.randn(samples_per_class, 2) * 0.8 + torch.tensor([-2.0, -2.0])

    # Class 1: points around the upper-right area.
    class1 = torch.randn(samples_per_class, 2) * 0.8 + torch.tensor([2.0, 2.0])

    x = torch.cat([class0, class1], dim=0)

    # BCEWithLogitsLoss expects float labels with shape [N, 1].
    y0 = torch.zeros(samples_per_class, 1)
    y1 = torch.ones(samples_per_class, 1)
    y = torch.cat([y0, y1], dim=0)

    return x, y


def train_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 200,
    learning_rate: float = 0.1,
) -> list[float]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_history: list[float] = []

    print_section("2. Training / 训练")

    for epoch in range(1, epochs + 1):
        # Forward: model outputs raw scores, also called logits.
        logits = model(x)

        # BCEWithLogitsLoss combines sigmoid + binary cross entropy.
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            accuracy = calculate_accuracy(model, x, y)
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {accuracy * 100:.2f}%"
            )

    return loss_history


def calculate_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        correct = (predictions == y).float().mean()

    return correct.item()


def save_plot(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_history: list[float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_min, x_max = x[:, 0].min().item() - 1, x[:, 0].max().item() + 1
    y_min, y_max = x[:, 1].min().item() - 1, x[:, 1].max().item() + 1

    xx, yy = torch.meshgrid(
        torch.linspace(x_min, x_max, 200),
        torch.linspace(y_min, y_max, 200),
        indexing="xy",
    )
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    with torch.no_grad():
        grid_logits = model(grid)
        grid_probabilities = torch.sigmoid(grid_logits).reshape(xx.shape)

    axes[0].contourf(
        xx.numpy(),
        yy.numpy(),
        grid_probabilities.numpy(),
        levels=20,
        cmap="coolwarm",
        alpha=0.35,
    )
    axes[0].contour(
        xx.numpy(),
        yy.numpy(),
        grid_probabilities.numpy(),
        levels=[0.5],
        colors="black",
        linewidths=2,
    )
    axes[0].scatter(
        x[:, 0].numpy(),
        x[:, 1].numpy(),
        c=y.squeeze().numpy(),
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.4,
    )
    axes[0].set_title("Binary Classification Boundary")
    axes[0].set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")

    axes[1].plot(loss_history, color="green")
    axes[1].set_title("Loss Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BCEWithLogitsLoss")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("PyTorch version:", torch.__version__)

    print_section("0. Generate Data / 生成数据")
    x, y = generate_data()

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("First 5 x values / 前 5 个 x:")
    print(x[:5])
    print("First 5 y labels / 前 5 个标签:")
    print(y[:5].squeeze())

    print_section("1. Build Model / 创建模型")

    # Two input features, one output logit.
    # Formula: logit = w1*x1 + w2*x2 + b
    model = nn.Linear(2, 1)

    print(model)
    print("Initial weight / 初始权重:")
    print(model.weight.data)
    print("Initial bias / 初始偏置:")
    print(model.bias.data)

    loss_history = train_model(model, x, y)

    print_section("3. Final Result / 最终结果")
    final_accuracy = calculate_accuracy(model, x, y)
    print("Learned weight / 学到的权重:")
    print(model.weight.data)
    print("Learned bias / 学到的偏置:")
    print(model.bias.data)
    print(f"Final accuracy / 最终准确率: {final_accuracy * 100:.2f}%")

    sample = torch.tensor([[1.5, 2.0]])
    with torch.no_grad():
        sample_logit = model(sample)
        sample_probability = torch.sigmoid(sample_logit)
        sample_prediction = (sample_probability >= 0.5).float()

    print_section("4. Predict One Point / 预测一个点")
    print("sample x:", sample)
    print("logit:", sample_logit.item())
    print("probability after sigmoid:", sample_probability.item())
    print("predicted class:", int(sample_prediction.item()))

    output_path = Path(__file__).with_name("classification_boundary.png")
    save_plot(model, x, y, loss_history, output_path)

    print_section("5. Saved Output / 保存结果")
    print("Figure saved to / 图片已保存到:")
    print(output_path)

    print_section("6. Summary / 小结")
    print("This experiment trains a logistic regression model for binary classification.")
    print("重点：模型输出 logit，sigmoid 把 logit 变成概率，0.5 是分类阈值。")


if __name__ == "__main__":
    main()
