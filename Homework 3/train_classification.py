# AI was used in generation of this code
"""
Training script for the Classifier model (Part 1).
Usage: python train_classification.py
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric
from homework.models import Classifier, save_model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader = load_data(
        "classification_data/train",
        transform_pipeline="aug",
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = load_data(
        "classification_data/val",
        transform_pipeline="default",
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Model, loss, optimizer
    model = Classifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_acc = 0.0
    metric = AccuracyMetric()

    for epoch in range(args.epochs):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        # ---- Validate ----
        model.eval()
        metric.reset()

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                preds = model.predict(images)
                metric.add(preds, labels)

        metrics = metric.compute()
        val_acc = metrics["accuracy"]

        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model)
            print(f"  -> Saved model (best acc: {best_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    train(args)
