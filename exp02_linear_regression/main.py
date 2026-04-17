from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


def print_section(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def generate_data() -> tuple[torch.Tensor, torch.Tensor]:
    # Synthetic data: y = 3x + 2 + noise
    torch.manual_seed(42)
    x = torch.linspace(-5, 5, 100).unsqueeze(1)
    noise = torch.randn(100, 1) * 0.8
    y = 3 * x + 2 + noise
    return x, y


def train_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 200,
    learning_rate: float = 0.05,
) -> list[float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_history: list[float] = []

    print_section("1. Training / 训练开始")

    for epoch in range(1, epochs + 1):
        # Forward pass: compute predictions
        predictions = model(x)

        # Compute loss
        loss = criterion(predictions, y)

        # Clear old gradients before backprop
        optimizer.zero_grad()

        # Compute gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        loss_history.append(loss.item())

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(x.numpy(), y.numpy(), label="Real data", alpha=0.7)
    axes[0].plot(
        x.numpy(),
        predictions.numpy(),
        color="red",
        linewidth=2,
        label="Predicted line",
    )
    axes[0].set_title("Linear Regression Fit")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()

    axes[1].plot(loss_history, color="green")
    axes[1].set_title("Loss Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")

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
    print(x[:5].squeeze())
    print("First 5 y values / 前 5 个 y:")
    print(y[:5].squeeze())

    print_section("1. Build Model / 创建模型")
    model = nn.Linear(1, 1)
    print(model)
    print("Initial weight / 初始权重:", model.weight.item())
    print("Initial bias / 初始偏置:", model.bias.item())

    loss_history = train_model(model, x, y)

    print_section("2. Final Parameters / 最终参数")
    learned_weight = model.weight.item()
    learned_bias = model.bias.item()
    print(f"Learned weight / 学到的权重: {learned_weight:.4f}")
    print(f"Learned bias / 学到的偏置: {learned_bias:.4f}")
    print("Target weight / 真实权重: 3.0000")
    print("Target bias / 真实偏置: 2.0000")

    with torch.no_grad():
        predictions = model(x)

    output_path = Path(__file__).with_name("linear_regression_fit.png")
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
    main()
