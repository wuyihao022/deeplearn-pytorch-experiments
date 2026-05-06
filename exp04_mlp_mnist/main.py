from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def print_section(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_dataloaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    data_dir = Path(__file__).parent / "data"

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 0.001,
) -> tuple[list[float], list[float]]:
    # 多分类任务的损失函数：衡量 10 个类别分数和真实标签之间的差距
    criterion = nn.CrossEntropyLoss()
    # Adam 优化器：根据梯度更新模型参数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 记录每一轮训练的平均 loss
    train_loss_history: list[float] = []
    # 记录每一轮测试集准确率
    test_accuracy_history: list[float] = []

    # 打印训练阶段标题
    print_section("2. Training / 训练")

    # 训练多个 epoch，每一轮都完整跑一次训练集和测试集
    for epoch in range(1, epochs + 1):
        # 训练一整轮，返回这一轮的平均 loss 和训练准确率
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        # 在测试集上评估当前模型，不更新参数
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        # 保存这一轮的训练 loss，方便后面画曲线
        train_loss_history.append(train_loss)
        # 保存这一轮的测试准确率，方便后面画曲线
        test_accuracy_history.append(test_accuracy)

        # 打印这一轮的训练和测试结果
        print(
            f"Epoch {epoch:2d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy * 100:.2f}%"
        )

    # 返回训练过程中的 loss 和准确率记录
    return train_loss_history, test_accuracy_history


def save_examples(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> None:
    model.eval()

    images, labels = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu()

    images = images.cpu()

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index].squeeze(), cmap="gray")
        axis.set_title(f"Pred: {predictions[index]} / True: {labels[index]}")
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_curves(
    train_loss_history: list[float],
    test_accuracy_history: list[float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(train_loss_history, marker="o")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropyLoss")

    axes[1].plot([accuracy * 100 for accuracy in test_accuracy_history], marker="o")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("PyTorch version:", torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print_section("0. Load Data / 加载数据")
    train_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))
    print("images shape:", images.shape)
    print("labels shape:", labels.shape)
    print("one image shape:", images[0].shape)
    print("first 10 labels:", labels[:10])

    print_section("1. Build Model / 创建模型")
    model = MLP().to(device)
    print(model)

    train_loss_history, test_accuracy_history = train_model(
        model,
        train_loader,
        test_loader,
        device,
    )

    criterion = nn.CrossEntropyLoss()
    final_test_loss, final_test_accuracy = evaluate(model, test_loader, criterion, device)

    print_section("3. Final Result / 最终结果")
    print(f"Final test loss: {final_test_loss:.4f}")
    print(f"Final test accuracy: {final_test_accuracy * 100:.2f}%")

    example_path = Path(__file__).with_name("mnist_predictions.png")
    curve_path = Path(__file__).with_name("training_curves.png")

    save_examples(model, test_loader, device, example_path)
    save_curves(train_loss_history, test_accuracy_history, curve_path)

    print_section("4. Saved Output / 保存结果")
    print("Prediction examples saved to:")
    print(example_path)
    print("Training curves saved to:")
    print(curve_path)

    print_section("5. Summary / 小结")
    print("This experiment trains an MLP to classify MNIST digits from 0 to 9.")
    print("重点：28x28 图片先展平成 784 个数，最后输出 10 个类别分数。")


if __name__ == "__main__":
    main()
