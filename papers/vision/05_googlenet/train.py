"""
Minimal training script for GoogLeNet on a CIFAR-10 subset.
Purpose: verify the architecture and auxiliary loss flow, not beat SOTA.

Runs ~5 epochs on 5000 images — finishes in a few minutes on CPU.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import GoogLeNet

EPOCHS     = 5
BATCH      = 64
LR         = 1e-3
NUM_TRAIN  = 5000   # subset size — change to len(dataset) for full run
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loaders() -> tuple[DataLoader, DataLoader]:
    # Resize to 224x224 so the spatial trace matches the architecture exactly
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_full = datasets.CIFAR10("./data", train=True,  download=True, transform=transform)
    val_full   = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(Subset(train_full, range(NUM_TRAIN)), batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(Subset(val_full,   range(1000)),      batch_size=BATCH, shuffle=False)

    return train_loader, val_loader


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        main, aux1, aux2 = model(images)

        # Total loss = main + 0.3 * aux1 + 0.3 * aux2  (from paper)
        loss = criterion(main, labels) \
             + 0.3 * criterion(aux1, labels) \
             + 0.3 * criterion(aux2, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += main.argmax(1).eq(labels).sum().item()
        total      += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        out  = model(images)   # eval mode returns only main logits
        loss = criterion(out, labels)

        total_loss += loss.item() * images.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += images.size(0)

    return total_loss / total, 100.0 * correct / total


def main() -> None:
    print(f"Device: {DEVICE} | Train samples: {NUM_TRAIN}")

    model     = GoogLeNet(num_classes=10, aux_logits=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print("-" * 52)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2f}%  {val_loss:>8.4f}  {val_acc:>6.2f}%")


if __name__ == "__main__":
    train_loader, val_loader = get_loaders()
    main()