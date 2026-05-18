from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from losses import reconstruction_errors, unpack_reconstruction


def compute_reconstruction_results(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    score_type: str = "mse",
    max_examples_per_class: int = 3,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, np.ndarray]]]]:
    model.eval()
    rows: list[dict] = []
    example_pool: dict[str, list[dict[str, np.ndarray]]] = {}

    with torch.no_grad():
        for images, labels, class_names, patient_ids, paths in loader:
            images = images.to(device, non_blocking=True)
            model_output = model(images)
            reconstructions = unpack_reconstruction(model_output)
            errors = reconstruction_errors(images, reconstructions, score_type)

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


def build_classwise_summary(results_df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
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
    if threshold is not None:
        detections = (
            results_df.assign(detected=(results_df["reconstruction_error"] > threshold).astype(int))
            .groupby("class_name", as_index=False)
            .agg(detected_count=("detected", "sum"))
        )
        summary = summary.merge(detections, on="class_name", how="left")
        summary["detected_count"] = summary["detected_count"].fillna(0).astype(int)
        summary["detection_rate"] = summary["detected_count"] / summary["sample_count"].clip(lower=1)
    return summary


def plot_training_loss(history: dict, save_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(history["train_loss"], label="Train", linewidth=2)
    axis.plot(history["val_loss"], label="Validation", linewidth=2)
    axis.set_title("Training vs Validation Objective")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
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


def _load_image_tensor(path: Path, image_size: int, device: torch.device, crop_mode: str = "none") -> torch.Tensor:
    from data_utils import build_transform
    from PIL import Image

    with Image.open(path) as image:
        tensor = build_transform(image_size, crop_mode=crop_mode)(image.convert("L")).unsqueeze(0)
    return tensor.to(device)


def _select_ranked_rows(results_df: pd.DataFrame, threshold: float) -> list[pd.Series]:
    selections: list[pd.Series] = []

    normal_df = results_df[results_df["class_name"] == "NORMAL"]
    pathology_df = results_df[results_df["label"] == 1]
    drusen_df = results_df[results_df["class_name"] == "DRUSEN"]

    candidates = [
        normal_df.sort_values("reconstruction_error", ascending=True).head(1),
        normal_df.sort_values("reconstruction_error", ascending=False).head(1),
        pathology_df.sort_values("reconstruction_error", ascending=False).head(1),
        pathology_df.sort_values("reconstruction_error", ascending=True).head(1),
        drusen_df[drusen_df["reconstruction_error"] <= threshold].sort_values("reconstruction_error", ascending=False).head(1),
    ]
    for frame in candidates:
        for _, row in frame.iterrows():
            if str(row["path"]) not in {str(item["path"]) for item in selections}:
                selections.append(row)
    return selections


def save_ranked_reconstruction_grid(
    model: torch.nn.Module,
    results_df: pd.DataFrame,
    threshold: float,
    image_size: int,
    device: torch.device,
    save_path: Path,
    crop_mode: str = "none",
) -> None:
    rows = _select_ranked_rows(results_df, threshold)
    if not rows:
        return

    model.eval()
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.8 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for row_index, row in enumerate(rows):
            image_tensor = _load_image_tensor(Path(row["path"]), image_size, device, crop_mode=crop_mode)
            reconstruction = unpack_reconstruction(model(image_tensor))
            original_np = image_tensor.cpu().numpy()[0, 0]
            reconstruction_np = reconstruction.cpu().numpy()[0, 0]
            residual_np = np.abs(original_np - reconstruction_np)
            predicted_label = "anomaly" if row["reconstruction_error"] > threshold else "normal"
            title_prefix = f"{row['class_name']} | score={row['reconstruction_error']:.5f} | pred={predicted_label}"

            for column_index, (title, image) in enumerate(
                [
                    ("Original", original_np),
                    ("Reconstruction", reconstruction_np),
                    ("Residual", residual_np),
                ]
            ):
                axes[row_index, column_index].imshow(image, cmap="gray")
                axes[row_index, column_index].set_title(f"{title_prefix}\n{title}", fontsize=8)
                axes[row_index, column_index].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max <= image_min:
        return np.zeros_like(image)
    return (image - image_min) / (image_max - image_min)


def _build_residual_overlay(original: np.ndarray, residual: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.stack([_normalize_image(original)] * 3, axis=-1)
    residual = np.asarray(residual, dtype=np.float32)
    scale = float(np.percentile(residual, 99))
    if scale <= 1e-8:
        heat_values = np.zeros_like(residual)
    else:
        heat_values = np.clip(residual / scale, 0.0, 1.0)
    heatmap = plt.get_cmap("magma")(heat_values)[..., :3]
    return np.clip((1.0 - alpha) * base + alpha * heatmap, 0.0, 1.0)


def _append_first_unique(
    selections: list[tuple[str, pd.Series]],
    used_paths: set[str],
    label: str,
    frame: pd.DataFrame,
) -> None:
    for _, row in frame.iterrows():
        path = str(row["path"])
        if path not in used_paths:
            selections.append((label, row))
            used_paths.add(path)
            return


def _select_overlay_rows(results_df: pd.DataFrame, threshold: float) -> list[tuple[str, pd.Series]]:
    selections: list[tuple[str, pd.Series]] = []
    used_paths: set[str] = set()

    normal_df = results_df[results_df["class_name"] == "NORMAL"]
    cnv_df = results_df[results_df["class_name"] == "CNV"]
    dme_df = results_df[results_df["class_name"] == "DME"]
    drusen_df = results_df[results_df["class_name"] == "DRUSEN"]

    _append_first_unique(
        selections,
        used_paths,
        "NORMAL true negative",
        normal_df[normal_df["reconstruction_error"] <= threshold].sort_values("reconstruction_error", ascending=True),
    )
    _append_first_unique(
        selections,
        used_paths,
        "NORMAL false positive",
        normal_df[normal_df["reconstruction_error"] > threshold].sort_values("reconstruction_error", ascending=False),
    )
    _append_first_unique(
        selections,
        used_paths,
        "CNV true positive",
        cnv_df[cnv_df["reconstruction_error"] > threshold].sort_values("reconstruction_error", ascending=False),
    )
    _append_first_unique(
        selections,
        used_paths,
        "DME true positive",
        dme_df[dme_df["reconstruction_error"] > threshold].sort_values("reconstruction_error", ascending=False),
    )
    _append_first_unique(
        selections,
        used_paths,
        "DRUSEN true positive",
        drusen_df[drusen_df["reconstruction_error"] > threshold].sort_values("reconstruction_error", ascending=False),
    )
    _append_first_unique(
        selections,
        used_paths,
        "DRUSEN false negative",
        drusen_df[drusen_df["reconstruction_error"] <= threshold].sort_values("reconstruction_error", ascending=False),
    )
    return selections


def _render_overlay_grid(
    model: torch.nn.Module,
    rows: list[tuple[str, pd.Series]],
    threshold: float,
    image_size: int,
    device: torch.device,
    save_path: Path,
    title: str,
    crop_mode: str = "none",
) -> None:
    if not rows:
        return

    model.eval()
    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 2.45 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for row_index, (selection_label, row) in enumerate(rows):
            image_tensor = _load_image_tensor(Path(row["path"]), image_size, device, crop_mode=crop_mode)
            reconstruction = unpack_reconstruction(model(image_tensor))
            original_np = image_tensor.cpu().numpy()[0, 0]
            reconstruction_np = reconstruction.cpu().numpy()[0, 0]
            residual_np = np.abs(original_np - reconstruction_np)
            overlay_np = _build_residual_overlay(original_np, residual_np)
            prediction = "anomaly" if row["reconstruction_error"] > threshold else "normal"
            row_title = f"{selection_label}\n{row['class_name']} | score={row['reconstruction_error']:.5f} | pred={prediction}"

            panels = [
                ("Original", original_np, "gray"),
                ("Reconstruction", reconstruction_np, "gray"),
                ("Residual", residual_np, "magma"),
                ("Residual heatmap overlay", overlay_np, None),
            ]
            for column_index, (panel_title, image, cmap) in enumerate(panels):
                axes[row_index, column_index].imshow(image, cmap=cmap)
                if row_index == 0:
                    axes[row_index, column_index].set_title(panel_title, fontsize=9)
                axes[row_index, column_index].axis("off")
            axes[row_index, 0].text(
                -0.08,
                0.5,
                row_title,
                transform=axes[row_index, 0].transAxes,
                fontsize=8,
                ha="right",
                va="center",
                clip_on=False,
            )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.18, 0, 1, 0.985))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_residual_heatmap_overlay_grid(
    model: torch.nn.Module,
    results_df: pd.DataFrame,
    threshold: float,
    image_size: int,
    device: torch.device,
    save_path: Path,
    crop_mode: str = "none",
) -> None:
    rows = _select_overlay_rows(results_df, threshold)
    _render_overlay_grid(
        model=model,
        rows=rows,
        threshold=threshold,
        image_size=image_size,
        device=device,
        save_path=save_path,
        title="Residual Heatmap Overlay Examples",
        crop_mode=crop_mode,
    )


def save_drusen_false_negative_artifacts(
    model: torch.nn.Module,
    results_df: pd.DataFrame,
    threshold: float,
    image_size: int,
    device: torch.device,
    figure_path: Path,
    csv_path: Path,
    max_examples: int = 6,
    crop_mode: str = "none",
) -> pd.DataFrame:
    drusen_false_negatives = results_df[
        (results_df["class_name"] == "DRUSEN")
        & (results_df["label"] == 1)
        & (results_df["reconstruction_error"] <= threshold)
    ].copy()
    drusen_false_negatives["threshold"] = threshold
    drusen_false_negatives["margin_below_threshold"] = threshold - drusen_false_negatives["reconstruction_error"]
    drusen_false_negatives = drusen_false_negatives.sort_values("reconstruction_error", ascending=False)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    drusen_false_negatives.to_csv(csv_path, index=False)

    selected_rows = [
        (f"DRUSEN false negative #{index + 1}", row)
        for index, (_, row) in enumerate(drusen_false_negatives.head(max_examples).iterrows())
    ]
    _render_overlay_grid(
        model=model,
        rows=selected_rows,
        threshold=threshold,
        image_size=image_size,
        device=device,
        save_path=figure_path,
        title="DRUSEN False Negative Examples",
        crop_mode=crop_mode,
    )
    return drusen_false_negatives


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
        f"  run_id             : {config.get('run_id')}\n"
        f"  model_type         : {config['model_type']}\n"
        f"  loss_type          : {config['loss_type']}\n"
        f"  data_root          : {config['data_root']}\n"
        f"  image_size         : {config['image_size']}\n"
        f"  crop_mode          : {config['crop_mode']}\n"
        f"  batch_size         : {config['batch_size']}\n"
        f"  latent_dim         : {config['latent_dim']}\n"
        f"  num_workers        : {config['num_workers']}\n"
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
