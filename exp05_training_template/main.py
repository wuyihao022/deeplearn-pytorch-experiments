import argparse
from pathlib import Path

import torch
from torch import nn

from dataset import build_dataloaders
from inference import save_prediction_examples
from model import MLP
from train import fit, validate
from utils import load_checkpoint, print_section, save_history_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 05 training template")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).parent
    data_dir = project_dir / "data"
    output_dir = project_dir / "outputs"
    checkpoint_path = project_dir / "checkpoints" / "best_mlp.pt"

    print_section("0. Setup")
    print("PyTorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print_section("1. Data")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
    )
    images, labels = next(iter(train_loader))
    print("train images shape:", images.shape)
    print("train labels shape:", labels.shape)
    print("first 10 labels:", labels[:10])

    print_section("2. Model")
    model = MLP(hidden_size=args.hidden_size).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print_section("3. Training")
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
    )

    print_section("4. Test Best Checkpoint")
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    print(f"Best checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best val accuracy: {checkpoint['best_val_accuracy'] * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    print_section("5. Save Outputs")
    curve_path = output_dir / "training_history.png"
    prediction_path = output_dir / "prediction_examples.png"
    save_history_plot(history, curve_path)
    save_prediction_examples(model, test_loader, device, prediction_path)
    print("Checkpoint saved to:", checkpoint_path)
    print("Training curves saved to:", curve_path)
    print("Prediction examples saved to:", prediction_path)

    print_section("6. Summary")
    print("This experiment turns the MNIST script into a reusable training template.")
    print("Main reusable parts: train_one_epoch, validate, predict, save_checkpoint.")


if __name__ == "__main__":
    main()
