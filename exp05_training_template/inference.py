from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader


def predict(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        predictions = logits.argmax(dim=1)

    return predictions.cpu()


def save_prediction_examples(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_path: Path,
    num_examples: int = 10,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images, labels = next(iter(data_loader))
    predictions = predict(model, images[:num_examples], device)

    rows = 2
    cols = num_examples // rows
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4))

    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index].squeeze(), cmap="gray")
        axis.set_title(f"Pred: {predictions[index]} / True: {labels[index]}")
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
