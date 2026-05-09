import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def print_section(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


@dataclass
class ExperimentConfig:
    name: str
    hidden_sizes: tuple[int, ...]
    dropout: float = 0.0
    weight_decay: float = 0.0
    early_stopping: bool = False
    patience: int = 3


class MLP(nn.Module):
    def __init__(self, hidden_sizes: tuple[int, ...], dropout: float = 0.0) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Flatten()]
        input_size = 28 * 28

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 06: overfitting and regularization on MNIST"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.ToTensor()

    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    full_test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    if train_size + val_size > len(full_train_dataset):
        raise ValueError("train_size + val_size is larger than MNIST train set.")
    if test_size > len(full_test_dataset):
        raise ValueError("test_size is larger than MNIST test set.")

    generator = torch.Generator().manual_seed(seed)
    unused_size = len(full_train_dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(
        full_train_dataset,
        [train_size, val_size, unused_size],
        generator=generator,
    )
    test_dataset = Subset(full_test_dataset, range(test_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


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
            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def run_experiment(
    config: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> dict[str, object]:
    print_section(config.name)
    model = MLP(config.hidden_sizes, config.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | train acc {train_acc * 100:5.2f}% | "
            f"val loss {val_loss:.4f} | val acc {val_acc * 100:5.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            config.early_stopping
            and epochs_without_improvement >= config.patience
        ):
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    overfit_gap = history["train_acc"][-1] - history["val_acc"][-1]

    print(
        f"Best epoch: {best_epoch} | "
        f"test loss {test_loss:.4f} | test acc {test_acc * 100:.2f}% | "
        f"train-val acc gap {overfit_gap * 100:.2f}%"
    )

    return {
        "config": config,
        "history": history,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "overfit_gap": overfit_gap,
    }


def save_comparison_plots(results: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for result in results:
        config = result["config"]
        history = result["history"]
        assert isinstance(config, ExperimentConfig)
        assert isinstance(history, dict)

        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(
            epochs,
            history["train_loss"],
            marker="o",
            linewidth=1.5,
            label=f"{config.name} train",
        )
        axes[0].plot(
            epochs,
            history["val_loss"],
            marker="x",
            linestyle="--",
            linewidth=1.5,
            label=f"{config.name} val",
        )
        axes[1].plot(
            epochs,
            [acc * 100 for acc in history["val_acc"]],
            marker="o",
            linewidth=1.5,
            label=config.name,
        )

    axes[0].set_title("Train Loss vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropyLoss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "regularization_comparison.png", dpi=150)
    plt.close(fig)


def save_summary_csv(results: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "name",
                "best_epoch",
                "final_train_acc",
                "final_val_acc",
                "train_val_acc_gap",
                "test_loss",
                "test_acc",
            ]
        )
        for result in results:
            config = result["config"]
            history = result["history"]
            assert isinstance(config, ExperimentConfig)
            assert isinstance(history, dict)
            writer.writerow(
                [
                    config.name,
                    result["best_epoch"],
                    f"{history['train_acc'][-1]:.4f}",
                    f"{history['val_acc'][-1]:.4f}",
                    f"{result['overfit_gap']:.4f}",
                    f"{result['test_loss']:.4f}",
                    f"{result['test_acc']:.4f}",
                ]
            )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    project_dir = Path(__file__).parent
    data_dir = project_dir / "data"
    output_dir = project_dir / "outputs"

    print_section("0. Setup")
    print("PyTorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print_section("1. Data")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("Small train set is intentional: it makes overfitting easier to see.")

    configs = [
        ExperimentConfig(
            name="baseline_big_model",
            hidden_sizes=(512, 256, 128),
        ),
        ExperimentConfig(
            name="dropout",
            hidden_sizes=(512, 256, 128),
            dropout=0.5,
        ),
        ExperimentConfig(
            name="weight_decay",
            hidden_sizes=(512, 256, 128),
            weight_decay=1e-4,
        ),
        ExperimentConfig(
            name="small_model",
            hidden_sizes=(128, 64),
        ),
        ExperimentConfig(
            name="early_stopping",
            hidden_sizes=(512, 256, 128),
            early_stopping=True,
            patience=3,
        ),
    ]

    print_section("2. Experiments")
    results = [
        run_experiment(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        for config in configs
    ]

    print_section("3. Save Outputs")
    save_comparison_plots(results, output_dir)
    save_summary_csv(results, output_dir / "summary.csv")
    print("Comparison plot saved to:", output_dir / "regularization_comparison.png")
    print("Summary table saved to:", output_dir / "summary.csv")

    print_section("4. Summary")
    for result in results:
        config = result["config"]
        assert isinstance(config, ExperimentConfig)
        print(
            f"{config.name:18s} | "
            f"test acc {result['test_acc'] * 100:5.2f}% | "
            f"train-val gap {result['overfit_gap'] * 100:5.2f}%"
        )
    print("If train accuracy keeps rising while validation loss rises, overfitting has started.")


if __name__ == "__main__":
    main()
