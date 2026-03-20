"""
Training script for the Detector model (Part 2).
Usage: python train_detection.py
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric
from homework.models import Detector, save_model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader = load_data(
        "drive_data/train",
        transform_pipeline="aug",
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = load_data(
        "drive_data/val",
        transform_pipeline="default",
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Model, losses, optimizer
    model = Detector().to(device)
    seg_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_score = -float("inf")
    metric = DetectionMetric()

    for epoch in range(args.epochs):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            image = batch["image"].to(device)
            track = batch["track"].to(device)
            depth = batch["depth"].to(device)

            logits, pred_depth = model(image)

            loss_seg = seg_loss_fn(logits, track)
            loss_depth = depth_loss_fn(pred_depth, depth)
            loss = loss_seg + args.depth_weight * loss_depth

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
            for batch in val_loader:
                image = batch["image"].to(device)
                track = batch["track"].to(device)
                depth = batch["depth"].to(device)

                pred, pred_depth = model.predict(image)
                metric.add(pred, track, pred_depth, depth)

        metrics = metric.compute()
        iou = metrics["iou"]
        acc = metrics["accuracy"]
        depth_err = metrics["abs_depth_error"]
        tp_depth_err = metrics["tp_depth_error"]

        # Use a combined score: higher is better
        score = iou - depth_err - tp_depth_err

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"IoU: {iou:.4f} | "
            f"Acc: {acc:.4f} | "
            f"Depth Err: {depth_err:.4f} | "
            f"TP Depth Err: {tp_depth_err:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if score > best_score:
            best_score = score
            save_model(model)
            print(f"  -> Saved model (IoU: {iou:.4f}, Depth: {depth_err:.4f}, TP Depth: {tp_depth_err:.4f})")

    print(f"\nTraining complete. Best combined score: {best_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    train(args)
