"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
# Used AI for assistance

import argparse

import torch
import torch.nn as nn

from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model


def train_mlp_planner(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_data(
        "drive_data/train",
        transform_pipeline="state_only",
        num_workers=2,
        batch_size=128,
        shuffle=True,
    )
    val_data = load_data(
        "drive_data/val",
        transform_pipeline="state_only",
        num_workers=2,
        batch_size=128,
        shuffle=False,
    )

    model = MLPPlanner().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_lat = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        count = 0

        for batch in train_data:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            pred = model(track_left, track_right)

            mask = waypoints_mask.unsqueeze(-1).float()
            loss = (torch.abs(pred - waypoints) * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        metric = PlannerMetric()
        model.eval()
        with torch.no_grad():
            for batch in val_data:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                pred = model(track_left, track_right)
                metric.add(pred, waypoints, waypoints_mask)

        metrics = metric.compute()
        val_loss = metrics["longitudinal_error"] + metrics["lateral_error"]
        scheduler.step(val_loss)

        if metrics["lateral_error"] < best_lat:
            best_lat = metrics["lateral_error"]
            save_model(model)

        print(
            f"[MLP] Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={total_loss/count:.4f}  "
            f"lon={metrics['longitudinal_error']:.4f}  "
            f"lat={metrics['lateral_error']:.4f}  "
            f"best_lat={best_lat:.4f}"
        )

    print("MLP Planner saved!")


def train_transformer_planner(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_data(
        "drive_data/train",
        transform_pipeline="state_only",
        num_workers=2,
        batch_size=128,
        shuffle=True,
    )
    val_data = load_data(
        "drive_data/val",
        transform_pipeline="state_only",
        num_workers=2,
        batch_size=128,
        shuffle=False,
    )

    model = TransformerPlanner().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_lat = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        count = 0

        for batch in train_data:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            pred = model(track_left, track_right)

            mask = waypoints_mask.unsqueeze(-1).float()
            loss = (torch.abs(pred - waypoints) * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        metric = PlannerMetric()
        model.eval()
        with torch.no_grad():
            for batch in val_data:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                pred = model(track_left, track_right)
                metric.add(pred, waypoints, waypoints_mask)

        metrics = metric.compute()
        val_loss = metrics["longitudinal_error"] + metrics["lateral_error"]
        scheduler.step(val_loss)

        if metrics["lateral_error"] < best_lat:
            best_lat = metrics["lateral_error"]
            save_model(model)

        print(
            f"[Transformer] Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={total_loss/count:.4f}  "
            f"lon={metrics['longitudinal_error']:.4f}  "
            f"lat={metrics['lateral_error']:.4f}  "
            f"best_lat={best_lat:.4f}"
        )

    print("Transformer Planner saved!")


def train_cnn_planner(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_data(
        "drive_data/train",
        transform_pipeline="default",
        num_workers=2,
        batch_size=64,
        shuffle=True,
    )
    val_data = load_data(
        "drive_data/val",
        transform_pipeline="default",
        num_workers=2,
        batch_size=64,
        shuffle=False,
    )

    model = CNNPlanner().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_lat = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        count = 0

        for batch in train_data:
            image = batch["image"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            pred = model(image)

            mask = waypoints_mask.unsqueeze(-1).float()
            loss = (torch.abs(pred - waypoints) * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        metric = PlannerMetric()
        model.eval()
        with torch.no_grad():
            for batch in val_data:
                image = batch["image"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                pred = model(image)
                metric.add(pred, waypoints, waypoints_mask)

        metrics = metric.compute()
        val_loss = metrics["longitudinal_error"] + metrics["lateral_error"]
        scheduler.step(val_loss)

        if metrics["lateral_error"] < best_lat:
            best_lat = metrics["lateral_error"]
            save_model(model)

        print(
            f"[CNN] Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={total_loss/count:.4f}  "
            f"lon={metrics['longitudinal_error']:.4f}  "
            f"lat={metrics['lateral_error']:.4f}  "
            f"best_lat={best_lat:.4f}"
        )

    print("CNN Planner saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "transformer", "cnn", "all"])
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if args.model in ("mlp", "all"):
        train_mlp_planner(args)
    if args.model in ("transformer", "all"):
        train_transformer_planner(args)
    if args.model in ("cnn", "all"):
        train_cnn_planner(args)
