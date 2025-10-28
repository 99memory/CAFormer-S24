"""Training script for the CAFormer-S24 architecture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from caformer import build_caformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CAFormer model")
    parser.add_argument("data_dir", type=Path, help="Root directory of the dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, help="Checkpoint path", default=None)
    return parser.parse_args()


def build_dataloaders(data_dir: Path, image_size: int, batch_size: int, workers: int) -> Dict[str, DataLoader]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_set = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_set = datasets.ImageFolder(data_dir / "val", transform=val_tfms)

    loaders = {
        "train": DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        ),
    }
    return loaders


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean()


def save_checkpoint(state: Dict[str, Any], output_dir: Path, is_best: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.pt"
    torch.save(state, ckpt_path)
    if is_best:
        torch.save(state, output_dir / "best.pt")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(logits, targets).item() * images.size(0)
    num_samples = len(loader.dataset)
    return {
        "loss": running_loss / num_samples,
        "acc": running_acc / num_samples,
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(logits, targets).item() * images.size(0)
    num_samples = len(loader.dataset)
    return {
        "loss": running_loss / num_samples,
        "acc": running_acc / num_samples,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loaders = build_dataloaders(args.data_dir, args.image_size, args.batch_size, args.workers)

    model = build_caformer(num_classes=args.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_acc = 0.0
    if args.resume and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0)
        best_acc = checkpoint.get("best_acc", 0.0)

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_stats = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        is_best = val_stats["acc"] > best_acc
        if is_best:
            best_acc = val_stats["acc"]

        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train": train_stats,
            "val": val_stats,
            "best_acc": best_acc,
            "config": vars(args),
        }
        save_checkpoint(state, args.output, is_best=is_best)

        log = {
            "epoch": epoch + 1,
            "train": train_stats,
            "val": val_stats,
            "lr": scheduler.get_last_lr()[0],
        }
        print(json.dumps(log))


if __name__ == "__main__":
    main()
