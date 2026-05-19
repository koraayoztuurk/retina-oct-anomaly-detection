from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


METRIC_COLUMNS = ["auroc", "f1", "recall", "precision", "accuracy", "fpr"]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def discover_experiment_dirs(experiments_root: Path, include_run_ids: set[str] | None = None) -> list[Path]:
    if not experiments_root.exists():
        return []
    return sorted(
        path
        for path in experiments_root.iterdir()
        if (path / "metrics" / "selected_threshold_metrics.json").exists()
        and (include_run_ids is None or path.name in include_run_ids)
    )


def load_experiment_summary(run_dir: Path) -> dict[str, Any]:
    metrics_dir = run_dir / "metrics"
    metrics = read_json(metrics_dir / "selected_threshold_metrics.json")
    config = read_json(metrics_dir / "run_config.json")
    history_path = metrics_dir / "training_history.json"
    history = read_json(history_path) if history_path.exists() else {}
    row = {
        "run_id": config.get("run_id") or run_dir.name,
        "model_type": config.get("model_type"),
        "use_batch_norm": config.get("use_batch_norm", False),
        "loss_type": config.get("loss_type"),
        "latent_dim": config.get("latent_dim"),
        "image_size": config.get("image_size"),
        "crop_mode": config.get("crop_mode", "none"),
        "batch_size": config.get("batch_size"),
        "default_percentile": config.get("default_percentile"),
        "threshold": metrics.get("threshold"),
        "best_epoch": history.get("best_epoch"),
        "best_val_loss": history.get("best_val_loss"),
        "training_time_sec": history.get("training_time_sec"),
    }
    for column in METRIC_COLUMNS + ["tn", "fp", "fn", "tp"]:
        row[column] = metrics.get(column)
    return row


def load_classwise(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "metrics" / "classwise_reconstruction_summary.csv"
    config = read_json(run_dir / "metrics" / "run_config.json")
    frame = pd.read_csv(path)
    frame.insert(0, "run_id", run_dir.name)
    frame.insert(1, "crop_mode", config.get("crop_mode", "none"))
    frame.insert(2, "batch_size", config.get("batch_size"))
    return frame


def load_thresholds(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "metrics" / "threshold_comparison.csv"
    frame = pd.read_csv(path)
    frame.insert(0, "run_id", run_dir.name)
    return frame


def load_baseline_outputs(baseline_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_dir = baseline_root / "metrics"
    metrics = read_json(metrics_dir / "selected_threshold_metrics.json")
    config = read_json(metrics_dir / "run_config.json")
    history_path = metrics_dir / "training_history.json"
    history = read_json(history_path) if history_path.exists() else {}
    comparison_df = pd.DataFrame(
        [
            {
                "run_id": "current_baseline",
                "model_type": config.get("model_type", "ae"),
                "use_batch_norm": config.get("use_batch_norm", False),
                "loss_type": config.get("loss_type", "mse"),
                "latent_dim": config.get("latent_dim", 128),
                "image_size": config.get("image_size", 128),
                "crop_mode": config.get("crop_mode", "none"),
                "batch_size": config.get("batch_size", 32),
                "default_percentile": config.get("default_percentile", 95),
                "threshold": metrics.get("threshold"),
                "best_epoch": history.get("best_epoch"),
                "best_val_loss": history.get("best_val_loss"),
                "training_time_sec": history.get("training_time_sec"),
                **{column: metrics.get(column) for column in METRIC_COLUMNS + ["tn", "fp", "fn", "tp"]},
            }
        ]
    )
    threshold_df = pd.read_csv(metrics_dir / "threshold_comparison.csv")
    threshold_df.insert(0, "run_id", "current_baseline")
    classwise_df = pd.read_csv(metrics_dir / "classwise_reconstruction_summary.csv")
    classwise_df.insert(0, "run_id", "current_baseline")
    if "detected_count" not in classwise_df.columns:
        test_df = pd.read_csv(metrics_dir / "test_reconstruction_errors.csv")
        detected = (
            test_df.assign(detected=(test_df["reconstruction_error"] > metrics["threshold"]).astype(int))
            .groupby("class_name", as_index=False)
            .agg(detected_count=("detected", "sum"))
        )
        classwise_df = classwise_df.merge(detected, on="class_name", how="left")
        classwise_df["detected_count"] = classwise_df["detected_count"].fillna(0).astype(int)
        classwise_df["detection_rate"] = classwise_df["detected_count"] / classwise_df["sample_count"].clip(lower=1)
    return comparison_df, threshold_df, classwise_df


def choose_best_run(comparison_df: pd.DataFrame) -> dict[str, Any]:
    if comparison_df.empty:
        return {}

    sorted_df = comparison_df.sort_values(["auroc", "f1", "recall"], ascending=False).reset_index(drop=True)
    best = sorted_df.iloc[0].to_dict()
    tied = comparison_df[comparison_df["auroc"] >= best["auroc"] - 0.005].copy()
    if len(tied) > 1:
        tied = tied.sort_values(["f1", "recall"], ascending=False)
        best = tied.iloc[0].to_dict()
    tied_f1 = comparison_df[
        (comparison_df["auroc"] >= best["auroc"] - 0.005)
        & (comparison_df["f1"] >= best["f1"] - 0.01)
    ].copy()
    if len(tied_f1) > 1:
        tied_f1 = tied_f1.sort_values(["recall", "f1", "auroc"], ascending=False)
        best = tied_f1.iloc[0].to_dict()
    return best


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: Path) -> None:
    if comparison_df.empty:
        return
    display = comparison_df.sort_values("run_id")
    x = np.arange(len(display))
    width = 0.24
    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(x - width, display["auroc"], width, label="AUROC")
    axis.bar(x, display["f1"], width, label="F1")
    axis.bar(x + width, display["recall"], width, label="Recall")
    axis.set_xticks(x, display["run_id"], rotation=25, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Final Experiment Model Comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_threshold_heatmap(threshold_df: pd.DataFrame, save_path: Path) -> None:
    if threshold_df.empty:
        return
    pivot = threshold_df.pivot(index="run_id", columns="percentile", values="f1")
    fig, axis = plt.subplots(figsize=(6.5, max(3.5, 0.5 * len(pivot))))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(pivot.columns)), [f"p{int(col)}" for col in pivot.columns])
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    axis.set_title("F1 by Threshold Percentile")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            axis.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_classwise_detection(classwise_df: pd.DataFrame, save_path: Path) -> None:
    if classwise_df.empty or "detection_rate" not in classwise_df.columns:
        return
    pivot = classwise_df.pivot(index="run_id", columns="class_name", values="detection_rate").fillna(0)
    fig, axis = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=axis)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Detection rate")
    axis.set_title("Class-wise Detection Rate at p95")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Class")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_roc_overlay_sources(sources: list[tuple[str, Path]], save_path: Path) -> None:
    if not sources:
        return
    fig, axis = plt.subplots(figsize=(6, 6))
    for label, test_path in sources:
        if not test_path.exists():
            continue
        frame = pd.read_csv(test_path)
        y_true = frame["label"].to_numpy()
        scores = frame["reconstruction_error"].to_numpy()
        fpr_values, tpr_values, _ = roc_curve(y_true, scores)
        auroc = roc_auc_score(y_true, scores)
        axis.plot(fpr_values, tpr_values, linewidth=1.8, label=f"{label} ({auroc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set_title("ROC Overlay")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_roc_overlay(run_dirs: list[Path], save_path: Path) -> None:
    sources = [(run_dir.name, run_dir / "metrics" / "test_reconstruction_errors.csv") for run_dir in run_dirs]
    plot_roc_overlay_sources(sources, save_path)


def aggregate_experiments(
    experiments_root: Path,
    comparison_root: Path,
    baseline_root: Path | None = None,
    include_run_ids: set[str] | None = None,
) -> dict[str, Any]:
    run_dirs = discover_experiment_dirs(experiments_root, include_run_ids=include_run_ids)
    comparison_root.mkdir(parents=True, exist_ok=True)

    if run_dirs:
        comparison_df = pd.DataFrame([load_experiment_summary(run_dir) for run_dir in run_dirs])
        threshold_df = pd.concat([load_thresholds(run_dir) for run_dir in run_dirs], ignore_index=True)
        classwise_df = pd.concat([load_classwise(run_dir) for run_dir in run_dirs], ignore_index=True)
        roc_sources = [(run_dir.name, run_dir / "metrics" / "test_reconstruction_errors.csv") for run_dir in run_dirs]
    elif baseline_root and (baseline_root / "metrics" / "selected_threshold_metrics.json").exists():
        comparison_df, threshold_df, classwise_df = load_baseline_outputs(baseline_root)
        roc_sources = [("current_baseline", baseline_root / "metrics" / "test_reconstruction_errors.csv")]
    else:
        comparison_df = pd.DataFrame()
        threshold_df = pd.DataFrame()
        classwise_df = pd.DataFrame()
        roc_sources = []
    best_run = choose_best_run(comparison_df)

    comparison_df.to_csv(comparison_root / "model_comparison.csv", index=False)
    threshold_df.to_csv(comparison_root / "threshold_comparison_all.csv", index=False)
    classwise_df.to_csv(comparison_root / "classwise_detection_counts.csv", index=False)
    save_json(best_run, comparison_root / "best_model.json")

    plot_model_comparison(comparison_df, comparison_root / "model_comparison.png")
    plot_threshold_heatmap(threshold_df, comparison_root / "threshold_heatmap.png")
    plot_classwise_detection(classwise_df, comparison_root / "classwise_detection_counts.png")
    plot_roc_overlay_sources(roc_sources, comparison_root / "roc_overlay.png")

    return {
        "run_count": len(comparison_df),
        "comparison_csv": str(comparison_root / "model_comparison.csv"),
        "best_run": best_run,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Retina OCT final experiment outputs.")
    parser.add_argument("--experiments-root", default="outputs/experiments")
    parser.add_argument("--comparison-root", default="outputs/comparison")
    parser.add_argument("--baseline-root", default="outputs")
    parser.add_argument("--only", nargs="*", default=None, help="Optional run_id values to include in the comparison.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_run_ids = set(args.only) if args.only else None
    summary = aggregate_experiments(
        Path(args.experiments_root),
        Path(args.comparison_root),
        Path(args.baseline_root),
        include_run_ids=include_run_ids,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
