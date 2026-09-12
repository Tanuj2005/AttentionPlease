"""
Training script for VGG on CIFAR-10 using VGGSmall.

Usage:
    python train.py
    python train.py --epochs 100 --lr 1e-1 --no-bn
    python train.py --wd 1e-4
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import VGGSmall


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VGG on CIFAR-10")
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--batch",   type=int,   default=128)
    parser.add_argument("--lr",      type=float, default=0.05)
    parser.add_argument("--wd",      type=float, default=5e-4,  help="weight decay")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-bn",   action="store_true",       help="disable batch norm")
    parser.add_argument("--save",    type=str,   default="vgg_cifar10.pth")
    return parser.parse_args()


def get_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR10("./data", train=True,  download=True, transform=train_transform)
    val_dataset   = datasets.CIFAR10("./data", train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += images.size(0)

    return total_loss / total, 100.0 * correct / total


def main() -> None:
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_bn = not args.no_bn
    model  = VGGSmall(batch_norm=use_bn, num_classes=10, dropout=args.dropout).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VGGSmall (BN={use_bn}) | Trainable parameters: {params:,}")

    train_loader, val_loader = get_loaders(args.batch)
    criterion = nn.CrossEntropyLoss()

    # SGD + momentum — same as the paper.
    # VGG was notoriously slow to train (2-3 weeks on 4 GPUs for ImageNet).
    # On CIFAR-10 with this smaller variant it's manageable.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.wd,
        nesterov=True,          # Nesterov momentum — slight improvement over vanilla
    )

    # MultiStep: halve LR at 50% and 75% of training — mirrors the paper's schedule
    milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_val_acc = 0.0
    print(f"LR milestones: {milestones}")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'Time':>6}")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr      = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.2f}%  {val_loss:>8.4f}  {val_acc:>6.2f}%  {lr:>8.6f}  {elapsed:>5.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save)
            print(f"         -> saved (best val acc: {best_val_acc:.2f}%)")

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.save}")


if __name__ == "__main__":
    main()