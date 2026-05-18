from __future__ import annotations

import time
from pathlib import Path

import torch

from losses import compute_loss
from utils import format_duration


def run_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    loss_type: str,
    beta: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_items = 0
    component_totals: dict[str, float] = {}

    for images, *_ in loader:
        images = images.to(device, non_blocking=True)
        model_output = model(images)
        loss, components = compute_loss(images, model_output, loss_type=loss_type, beta=beta)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        for name, value in components.items():
            component_totals[name] = component_totals.get(name, 0.0) + value * batch_size

    average_components = {
        name: value / max(total_items, 1)
        for name, value in component_totals.items()
    }
    return total_loss / max(total_items, 1), average_components


def train_autoencoder(model, train_loader, val_loader, device, config, checkpoint_path: Path) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_components": [],
        "val_components": [],
        "best_epoch": None,
        "best_val_loss": None,
        "training_time_sec": None,
    }

    start_time = time.time()
    loss_type = config["loss_type"]
    beta = float(config["beta"])
    print(f"[INFO] Training {config['model_type']} model from scratch")
    print(f"[INFO] Loss type                : {loss_type}")

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_components = run_epoch(
            model,
            train_loader,
            device,
            loss_type=loss_type,
            beta=beta,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_loss, val_components = run_epoch(
                model,
                val_loader,
                device,
                loss_type=loss_type,
                beta=beta,
                optimizer=None,
            )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_components"].append(train_components)
        history["val_components"].append(val_components)
        print(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"[INFO] Early stopping triggered at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["training_time_sec"] = round(elapsed, 2)

    print(f"[INFO] Best epoch             : {best_epoch}")
    print(f"[INFO] Best val loss          : {best_val_loss:.6f}")
    print(f"[INFO] Training duration      : {format_duration(elapsed)}")
    return history
