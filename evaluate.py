from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def compute_reconstruction_results(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_examples_per_class: int = 3,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, np.ndarray]]]]:
    model.eval()
    rows: list[dict] = []
    example_pool: dict[str, list[dict[str, np.ndarray]]] = {}

    with torch.no_grad():
        for images, labels, class_names, patient_ids, paths in loader:
            images = images.to(device, non_blocking=True)
            reconstructions = model(images)
            errors = torch.mean((images - reconstructions) ** 2, dim=(1, 2, 3))

            images_cpu = images.cpu().numpy()
            recon_cpu = reconstructions.cpu().numpy()
            errors_cpu = errors.cpu().numpy()

            for index in range(len(paths)):
                class_name = str(class_names[index])
                rows.append(
                    {
                        "path": str(paths[index]),
                        "patient_id": str(patient_ids[index]),
                        "class_name": class_name,
                        "label": int(labels[index]),
                        "reconstruction_error": float(errors_cpu[index]),
                    }
                )

                class_examples = example_pool.setdefault(class_name, [])
                if len(class_examples) < max_examples_per_class:
                    original = images_cpu[index, 0]
                    reconstruction = recon_cpu[index, 0]
                    residual = np.abs(original - reconstruction)
                    class_examples.append(
                        {
                            "original": original,
                            "reconstruction": reconstruction,
                            "residual": residual,
                            "path": str(paths[index]),
                        }
                    )

    return pd.DataFrame(rows), example_pool


def compute_thresholds(val_errors: np.ndarray, percentiles: list[int]) -> dict[int, float]:
    return {percentile: float(np.percentile(val_errors, percentile)) for percentile in percentiles}


def evaluate_binary_metrics(results_df: pd.DataFrame, threshold: float) -> dict:
    y_true = results_df["label"].to_numpy()
    scores = results_df["reconstruction_error"].to_numpy()
    y_pred = (scores > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold": threshold,
        "auroc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_threshold_table(results_df: pd.DataFrame, thresholds: dict[int, float]) -> pd.DataFrame:
    rows = []
    for percentile, threshold in sorted(thresholds.items()):
        metrics = evaluate_binary_metrics(results_df, threshold)
        rows.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
                "auroc": metrics["auroc"],
            }
        )
    return pd.DataFrame(rows)


def build_classwise_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby("class_name", as_index=False)
        .agg(
            sample_count=("path", "count"),
            patient_count=("patient_id", "nunique"),
            mean_reconstruction_error=("reconstruction_error", "mean"),
            std_reconstruction_error=("reconstruction_error", "std"),
        )
        .sort_values("class_name")
        .reset_index(drop=True)
    )
    summary["std_reconstruction_error"] = summary["std_reconstruction_error"].fillna(0.0)
    return summary


def plot_training_loss(history: dict, save_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(history["train_loss"], label="Train", linewidth=2)
    axis.plot(history["val_loss"], label="Validation", linewidth=2)
    axis.set_title("Training vs Validation Reconstruction Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE Loss")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_val_error_histogram(val_df: pd.DataFrame, thresholds: dict[int, float], save_path: Path) -> None:
    errors = val_df["reconstruction_error"].to_numpy()
    upper_limit = max(np.percentile(errors, 99.5), max(thresholds.values()) * 1.1)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(errors[errors <= upper_limit], bins=50, color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.2)
    for percentile, threshold in sorted(thresholds.items()):
        axis.axvline(threshold, linestyle="--", linewidth=1.7, label=f"p{percentile}: {threshold:.5f}")
    axis.set_title("Validation Normal Reconstruction Error Distribution")
    axis.set_xlabel("Reconstruction error")
    axis.set_ylabel("Count")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_test_error_distribution(test_df: pd.DataFrame, threshold: float, save_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for class_name, color in [
        ("NORMAL", "steelblue"),
        ("CNV", "salmon"),
        ("DME", "goldenrod"),
        ("DRUSEN", "mediumseagreen"),
    ]:
        class_values = test_df.loc[test_df["class_name"] == class_name, "reconstruction_error"].to_numpy()
        if len(class_values) == 0:
            continue
        axis.hist(class_values, bins=40, alpha=0.5, label=class_name, density=True, color=color)

    axis.axvline(threshold, color="black", linestyle="--", linewidth=1.8, label=f"Threshold {threshold:.5f}")
    axis.set_title("Test Reconstruction Error by Class")
    axis.set_xlabel("Reconstruction error")
    axis.set_ylabel("Density")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_roc_curve(test_df: pd.DataFrame, save_path: Path) -> None:
    y_true = test_df["label"].to_numpy()
    scores = test_df["reconstruction_error"].to_numpy()
    fpr_values, tpr_values, _ = roc_curve(y_true, scores)
    auroc = roc_auc_score(y_true, scores)

    fig, axis = plt.subplots(figsize=(5.5, 5.5))
    axis.plot(fpr_values, tpr_values, color="darkorange", linewidth=2, label=f"AUROC={auroc:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set_title("ROC Curve")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(metrics: dict, save_path: Path) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, axis = plt.subplots(figsize=(5.5, 5))
    image = axis.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black", fontsize=12)

    axis.set_xticks([0, 1], ["Pred Normal", "Pred Pathology"])
    axis.set_yticks([0, 1], ["True Normal", "True Pathology"])
    axis.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_classwise_error_summary(classwise_df: pd.DataFrame, save_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    colors = ["steelblue", "salmon", "goldenrod", "mediumseagreen"][: len(classwise_df)]
    axis.bar(
        classwise_df["class_name"],
        classwise_df["mean_reconstruction_error"],
        yerr=classwise_df["std_reconstruction_error"],
        capsize=6,
        color=colors,
    )
    axis.set_title("Mean Reconstruction Error by Test Class")
    axis.set_xlabel("Class")
    axis.set_ylabel("Mean reconstruction error")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_reconstruction_grid(example_pool: dict[str, list[dict[str, np.ndarray]]], save_path: Path) -> None:
    rows = []
    for class_name in ["NORMAL", "CNV", "DME", "DRUSEN"]:
        rows.extend((class_name, item) for item in example_pool.get(class_name, [])[:1])

    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, (class_name, item) in enumerate(rows):
        for column_index, (title, image) in enumerate(
            [
                (f"{class_name} original", item["original"]),
                (f"{class_name} recon", item["reconstruction"]),
                (f"{class_name} residual", item["residual"]),
            ]
        ):
            axes[row_index, column_index].imshow(image, cmap="gray")
            axes[row_index, column_index].set_title(title)
            axes[row_index, column_index].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def build_summary_text(
    config: dict,
    dataset_summary: pd.DataFrame,
    history: dict,
    metrics: dict,
    thresholds: dict[int, float],
    classwise_df: pd.DataFrame,
) -> str:
    threshold_lines = "\n".join(f"  p{percentile}: {value:.6f}" for percentile, value in sorted(thresholds.items()))
    classwise_lines = classwise_df.to_string(index=False)
    dataset_lines = dataset_summary.to_string(index=False)
    return (
        "Retina OCT anomaly detection experiment summary\n"
        "===============================================\n\n"
        "Configuration\n"
        f"  data_root          : {config['data_root']}\n"
        f"  image_size         : {config['image_size']}\n"
        f"  batch_size         : {config['batch_size']}\n"
        f"  latent_dim         : {config['latent_dim']}\n"
        f"  default_percentile : p{config['default_percentile']}\n\n"
        "Dataset summary\n"
        f"{dataset_lines}\n\n"
        "Training summary\n"
        f"  best_epoch         : {history['best_epoch']}\n"
        f"  best_val_loss      : {history['best_val_loss']:.6f}\n"
        f"  training_time_sec  : {history['training_time_sec']}\n\n"
        "Thresholds\n"
        f"{threshold_lines}\n\n"
        "Selected-threshold metrics\n"
        f"  AUROC              : {metrics['auroc']:.4f}\n"
        f"  Accuracy           : {metrics['accuracy']:.4f}\n"
        f"  Precision          : {metrics['precision']:.4f}\n"
        f"  Recall             : {metrics['recall']:.4f}\n"
        f"  F1                 : {metrics['f1']:.4f}\n"
        f"  FPR                : {metrics['fpr']:.4f}\n"
        f"  Confusion          : TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}\n\n"
        "Class-wise reconstruction summary\n"
        f"{classwise_lines}\n"
    )
