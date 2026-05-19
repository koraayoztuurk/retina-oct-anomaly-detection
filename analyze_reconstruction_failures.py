from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from evaluate import save_drusen_false_negative_artifacts, save_residual_heatmap_overlay_grid
from model import build_model
from utils import get_device, save_json


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate residual heatmap and failure analysis artifacts for a completed run.")
    parser.add_argument("--run-id", default="ae_mse_l256", help="Experiment id under outputs/experiments.")
    parser.add_argument("--experiments-root", default="outputs/experiments")
    parser.add_argument("--checkpoint-name", default="best_autoencoder.pt")
    parser.add_argument("--crop-mode", default=None, choices=["none", "content", "border", "retina_margin"], help="Optional crop mode override.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold override.")
    parser.add_argument("--max-drusen-examples", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.experiments_root) / args.run_id
    metrics_dir = run_dir / "metrics"
    recon_dir = run_dir / "reconstructions"
    checkpoint_path = run_dir / "saved_models" / args.checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = read_json(metrics_dir / "run_config.json")
    metrics = read_json(metrics_dir / "selected_threshold_metrics.json")
    threshold = float(args.threshold if args.threshold is not None else metrics["threshold"])
    crop_mode = args.crop_mode or config.get("crop_mode", "none")
    test_results = pd.read_csv(metrics_dir / "test_reconstruction_errors.csv")

    device = get_device()
    model = build_model(
        model_type=config["model_type"],
        latent_dim=int(config["latent_dim"]),
        image_size=int(config["image_size"]),
        use_batch_norm=bool(config.get("use_batch_norm", False)),
    ).to(device)
    model.load_state_dict(load_state_dict(checkpoint_path, device))

    recon_dir.mkdir(parents=True, exist_ok=True)
    residual_overlay_path = recon_dir / "residual_heatmap_overlay_examples.png"
    drusen_figure_path = recon_dir / "drusen_false_negative_examples.png"
    drusen_csv_path = metrics_dir / "drusen_false_negative_examples.csv"

    save_residual_heatmap_overlay_grid(
        model=model,
        results_df=test_results,
        threshold=threshold,
        image_size=int(config["image_size"]),
        device=device,
        save_path=residual_overlay_path,
        crop_mode=crop_mode,
    )
    drusen_false_negatives = save_drusen_false_negative_artifacts(
        model=model,
        results_df=test_results,
        threshold=threshold,
        image_size=int(config["image_size"]),
        device=device,
        figure_path=drusen_figure_path,
        csv_path=drusen_csv_path,
        max_examples=args.max_drusen_examples,
        crop_mode=crop_mode,
    )

    summary = {
        "run_id": args.run_id,
        "threshold": threshold,
        "crop_mode": crop_mode,
        "residual_heatmap_overlay": str(residual_overlay_path),
        "drusen_false_negative_figure": str(drusen_figure_path),
        "drusen_false_negative_csv": str(drusen_csv_path),
        "drusen_false_negative_count": int(len(drusen_false_negatives)),
    }
    save_json(summary, metrics_dir / "failure_analysis_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
