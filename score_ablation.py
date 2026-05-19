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
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from data_utils import build_transform, prepare_datasets
from losses import ssim_per_sample, unpack_reconstruction
from model import build_model
from utils import get_device, save_json, set_seed


BASE_SCORE_MODES = [
    "mse",
    "l1",
    "ssim_error",
    "mse_ssim_score",
    "retina_band_mse",
    "retina_weighted_mse",
    "topk_mse_5",
    "topk_mse_10",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_retina_band_mask(images: torch.Tensor, margin_ratio: float = 0.12) -> torch.Tensor:
    batch_size, _, height, width = images.shape
    x_min = int(width * 0.06)
    x_max = max(x_min + 1, int(width * 0.94))
    center = images[:, :, :, x_min:x_max].squeeze(1)
    row_p95 = torch.quantile(center, 0.95, dim=2)
    row_std = center.std(dim=2)
    row_score = row_p95 + 0.5 * row_std

    dynamic_threshold = torch.maximum(
        torch.full_like(row_score[:, 0], 0.12),
        row_score.max(dim=1).values * 0.35,
    )
    active = row_score >= dynamic_threshold[:, None]
    smooth_kernel = torch.ones((1, 1, 5), device=images.device)
    active_counts = F.conv1d(active.float().unsqueeze(1), smooth_kernel, padding=2).squeeze(1)
    active = active_counts >= 2

    masks = torch.zeros_like(images)
    margin = max(4, int(height * margin_ratio))
    min_band_height = int(height * 0.35)
    for index in range(batch_size):
        rows = torch.where(active[index])[0]
        if rows.numel() == 0:
            masks[index] = 1.0
            continue
        y_min = max(0, int(rows[0].item()) - margin)
        y_max = min(height, int(rows[-1].item()) + margin + 1)
        if y_max - y_min < min_band_height:
            masks[index] = 1.0
        else:
            masks[index, :, y_min:y_max, :] = 1.0
    return masks


def compute_score_batch(images: torch.Tensor, reconstructions: torch.Tensor, mode: str) -> torch.Tensor:
    residual = images - reconstructions
    squared = residual.pow(2)
    absolute = residual.abs()

    if mode == "mse":
        return squared.mean(dim=(1, 2, 3))
    if mode == "l1":
        return absolute.mean(dim=(1, 2, 3))
    if mode == "ssim_error":
        return 1.0 - ssim_per_sample(images, reconstructions)
    if mode == "mse_ssim_score":
        mse = squared.mean(dim=(1, 2, 3))
        structural_error = 1.0 - ssim_per_sample(images, reconstructions)
        return 0.8 * mse + 0.2 * structural_error
    if mode == "retina_band_mse":
        mask = build_retina_band_mask(images)
        return (squared * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    if mode == "retina_weighted_mse":
        mask = build_retina_band_mask(images)
        weights = 0.25 + 0.75 * mask
        return (squared * weights).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3)).clamp_min(1.0)
    if mode.startswith("topk_mse_"):
        percent = float(mode.rsplit("_", 1)[-1])
        flat = squared.flatten(start_dim=1)
        k = max(1, int(flat.size(1) * percent / 100.0))
        return torch.topk(flat, k=k, dim=1).values.mean(dim=1)

    raise ValueError(f"Unsupported score mode: {mode}")


def compute_score_frame(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    score_modes: list[str],
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, labels, class_names, patient_ids, paths in loader:
            images = images.to(device, non_blocking=True)
            reconstructions = unpack_reconstruction(model(images))
            batch_scores = {
                mode: compute_score_batch(images, reconstructions, mode).detach().cpu().numpy()
                for mode in score_modes
            }
            for index, path in enumerate(paths):
                row = {
                    "path": str(path),
                    "patient_id": str(patient_ids[index]),
                    "class_name": str(class_names[index]),
                    "label": int(labels[index]),
                }
                for mode in score_modes:
                    row[mode] = float(batch_scores[mode][index])
                rows.append(row)
    return pd.DataFrame(rows)


def add_ensemble_scores(val_df: pd.DataFrame, test_df: pd.DataFrame, base_modes: list[str]) -> list[str]:
    ensemble_specs = {
        "ensemble_mse_l1": ["mse", "l1"],
        "ensemble_mse_l1_ssim": ["mse", "l1", "ssim_error"],
        "ensemble_mse_retina": ["mse", "retina_band_mse"],
        "ensemble_mse_ssim_topk": ["mse", "ssim_error", "topk_mse_5"],
        "ensemble_l1_ssim_topk": ["l1", "ssim_error", "topk_mse_5"],
        "ensemble_mse_l1_retina": ["mse", "l1", "retina_band_mse"],
        "ensemble_all_base": base_modes,
    }
    created: list[str] = []
    for ensemble_name, modes in ensemble_specs.items():
        available_modes = [mode for mode in modes if mode in val_df.columns]
        if len(available_modes) < 2:
            continue
        val_parts = []
        test_parts = []
        for mode in available_modes:
            median = float(val_df[mode].median())
            iqr = float(val_df[mode].quantile(0.75) - val_df[mode].quantile(0.25))
            scale = iqr if iqr > 1e-12 else float(val_df[mode].std() or 1.0)
            val_parts.append((val_df[mode] - median) / scale)
            test_parts.append((test_df[mode] - median) / scale)
        val_df[ensemble_name] = np.vstack(val_parts).mean(axis=0)
        test_df[ensemble_name] = np.vstack(test_parts).mean(axis=0)
        created.append(ensemble_name)
    return created


def evaluate_scores(test_df: pd.DataFrame, score_column: str, threshold: float) -> dict[str, Any]:
    y_true = test_df["label"].to_numpy()
    scores = test_df[score_column].to_numpy()
    y_pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "score_mode": score_column,
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn) if (fp + tn) else 0.0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_arrays(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auroc = float("nan")
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, scores))
    return {
        "auroc": auroc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn) if (fp + tn) else 0.0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_comparison_tables(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_modes: list[str],
    threshold_percentiles: list[int],
    default_percentile: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold_rows = []
    selected_rows = []
    classwise_rows = []

    for mode in score_modes:
        thresholds = {
            percentile: float(np.percentile(val_df[mode].to_numpy(), percentile))
            for percentile in threshold_percentiles
        }
        for percentile, threshold in thresholds.items():
            metrics = evaluate_scores(test_df, mode, threshold)
            threshold_rows.append({"percentile": percentile, **metrics})
            if percentile == default_percentile:
                selected_rows.append(metrics)
                detected = test_df.assign(detected=(test_df[mode] > threshold).astype(int))
                summary = (
                    detected.groupby("class_name", as_index=False)
                    .agg(
                        sample_count=("path", "count"),
                        patient_count=("patient_id", "nunique"),
                        mean_score=(mode, "mean"),
                        std_score=(mode, "std"),
                        detected_count=("detected", "sum"),
                    )
                    .sort_values("class_name")
                )
                summary["detection_rate"] = summary["detected_count"] / summary["sample_count"].clip(lower=1)
                summary.insert(0, "score_mode", mode)
                classwise_rows.append(summary)

    return (
        pd.DataFrame(selected_rows).sort_values(["auroc", "f1", "recall"], ascending=False),
        pd.DataFrame(threshold_rows),
        pd.concat(classwise_rows, ignore_index=True) if classwise_rows else pd.DataFrame(),
    )


def aggregate_patient_scores(df: pd.DataFrame, score_modes: list[str], aggregation: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patient_id, group in df.groupby("patient_id", sort=True):
        class_names = sorted(group["class_name"].unique())
        if len(class_names) > 1:
            raise ValueError(
                "Patient-level aggregation found mixed classes for "
                f"patient_id={patient_id}: {class_names}. Patient IDs must be class-aware."
            )
        row: dict[str, Any] = {
            "path": f"patient:{patient_id}",
            "patient_id": patient_id,
            "class_name": class_names[0],
            "label": int(group["label"].max()),
            "sample_count": int(len(group)),
        }
        for mode in score_modes:
            values = group[mode].to_numpy(dtype=float)
            if aggregation == "mean":
                score = float(values.mean())
            elif aggregation == "max":
                score = float(values.max())
            elif aggregation == "top2_mean":
                top_k = min(2, len(values))
                score = float(np.sort(values)[-top_k:].mean())
            else:
                raise ValueError(f"Unsupported patient aggregation: {aggregation}")
            row[mode] = score
        rows.append(row)
    return pd.DataFrame(rows)


def build_patient_level_tables(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_modes: list[str],
    threshold_percentiles: list[int],
    default_percentile: int,
    aggregations: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected_rows = []
    threshold_rows = []
    classwise_rows = []

    for aggregation in aggregations:
        val_patient_df = aggregate_patient_scores(val_df, score_modes, aggregation)
        test_patient_df = aggregate_patient_scores(test_df, score_modes, aggregation)
        for mode in score_modes:
            thresholds = {
                percentile: float(np.percentile(val_patient_df[mode].to_numpy(), percentile))
                for percentile in threshold_percentiles
            }
            for percentile, threshold in thresholds.items():
                metrics = evaluate_scores(test_patient_df, mode, threshold)
                row = {"patient_aggregation": aggregation, "percentile": percentile, **metrics}
                threshold_rows.append(row)
                if percentile == default_percentile:
                    selected_rows.append({"patient_aggregation": aggregation, **metrics})
                    detected = test_patient_df.assign(detected=(test_patient_df[mode] > threshold).astype(int))
                    summary = (
                        detected.groupby("class_name", as_index=False)
                        .agg(
                            patient_count=("patient_id", "count"),
                            image_count=("sample_count", "sum"),
                            mean_score=(mode, "mean"),
                            std_score=(mode, "std"),
                            detected_count=("detected", "sum"),
                        )
                        .sort_values("class_name")
                    )
                    summary["detection_rate"] = summary["detected_count"] / summary["patient_count"].clip(lower=1)
                    summary.insert(0, "score_mode", mode)
                    summary.insert(0, "patient_aggregation", aggregation)
                    classwise_rows.append(summary)

    selected_df = pd.DataFrame(selected_rows).sort_values(["auroc", "f1", "recall"], ascending=False)
    best_patient = choose_best_score(selected_df.drop(columns=["patient_aggregation"], errors="ignore"))
    if best_patient:
        matching = selected_df[
            (selected_df["score_mode"] == best_patient["score_mode"])
            & (selected_df["auroc"] == best_patient["auroc"])
            & (selected_df["f1"] == best_patient["f1"])
        ]
        if not matching.empty:
            best_patient = matching.iloc[0].to_dict()
    return (
        selected_df,
        pd.DataFrame(threshold_rows),
        pd.concat(classwise_rows, ignore_index=True) if classwise_rows else pd.DataFrame(),
        best_patient,
    )


def build_bootstrap_ci(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_modes: list[str],
    default_percentile: int,
    seed: int,
    iterations: int,
    patient_aggregations: list[str],
) -> pd.DataFrame:
    if iterations <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    levels = [("image", val_df, test_df)]
    levels.extend(
        (
            f"patient_{aggregation}",
            aggregate_patient_scores(val_df, score_modes, aggregation),
            aggregate_patient_scores(test_df, score_modes, aggregation),
        )
        for aggregation in patient_aggregations
    )
    metric_names = ["auroc", "accuracy", "precision", "recall", "f1", "fpr"]

    for level_name, level_val_df, level_test_df in levels:
        y_true = level_test_df["label"].to_numpy(dtype=int)
        sample_count = len(level_test_df)
        for mode in score_modes:
            threshold = float(np.percentile(level_val_df[mode].to_numpy(dtype=float), default_percentile))
            scores = level_test_df[mode].to_numpy(dtype=float)
            point = evaluate_arrays(y_true, scores, threshold)
            bootstrap_values: dict[str, list[float]] = {metric: [] for metric in metric_names}

            for _ in range(iterations):
                indices = rng.integers(0, sample_count, size=sample_count)
                sampled_y = y_true[indices]
                sampled_scores = scores[indices]
                sampled_metrics = evaluate_arrays(sampled_y, sampled_scores, threshold)
                for metric in metric_names:
                    value = sampled_metrics[metric]
                    if not np.isnan(value):
                        bootstrap_values[metric].append(float(value))

            for metric in metric_names:
                values = np.asarray(bootstrap_values[metric], dtype=float)
                if values.size == 0:
                    ci_low = float("nan")
                    ci_high = float("nan")
                    std = float("nan")
                else:
                    ci_low = float(np.percentile(values, 2.5))
                    ci_high = float(np.percentile(values, 97.5))
                    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
                rows.append(
                    {
                        "level": level_name,
                        "score_mode": mode,
                        "metric": metric,
                        "point": point[metric],
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "std": std,
                        "iterations": iterations,
                    }
                )

    return pd.DataFrame(rows)


def choose_best_score(comparison_df: pd.DataFrame) -> dict[str, Any]:
    if comparison_df.empty:
        return {}
    sorted_df = comparison_df.sort_values(["auroc", "f1", "recall"], ascending=False).reset_index(drop=True)
    best = sorted_df.iloc[0].to_dict()
    tied = comparison_df[comparison_df["auroc"] >= best["auroc"] - 0.005].copy()
    if len(tied) > 1:
        best = tied.sort_values(["f1", "recall"], ascending=False).iloc[0].to_dict()
    tied_f1 = comparison_df[
        (comparison_df["auroc"] >= best["auroc"] - 0.005)
        & (comparison_df["f1"] >= best["f1"] - 0.01)
    ].copy()
    if len(tied_f1) > 1:
        best = tied_f1.sort_values(["recall", "f1", "auroc"], ascending=False).iloc[0].to_dict()
    return best


def plot_score_comparison(comparison_df: pd.DataFrame, save_path: Path) -> None:
    if comparison_df.empty:
        return
    display = comparison_df.sort_values("auroc", ascending=False)
    x = np.arange(len(display))
    width = 0.24
    fig, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar(x - width, display["auroc"], width, label="AUROC")
    axis.bar(x, display["f1"], width, label="F1")
    axis.bar(x + width, display["recall"], width, label="Recall")
    axis.set_xticks(x, display["score_mode"], rotation=30, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Anomaly Score Ablation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_roc_overlay(test_df: pd.DataFrame, comparison_df: pd.DataFrame, save_path: Path, top_n: int = 6) -> None:
    if comparison_df.empty:
        return
    selected_modes = comparison_df.sort_values("auroc", ascending=False)["score_mode"].head(top_n).tolist()
    y_true = test_df["label"].to_numpy()
    fig, axis = plt.subplots(figsize=(6, 6))
    for mode in selected_modes:
        scores = test_df[mode].to_numpy()
        fpr_values, tpr_values, _ = roc_curve(y_true, scores)
        auroc = roc_auc_score(y_true, scores)
        axis.plot(fpr_values, tpr_values, linewidth=1.8, label=f"{mode} ({auroc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set_title("Score Ablation ROC Overlay")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max <= image_min:
        return np.zeros_like(image)
    return (image - image_min) / (image_max - image_min)


def build_topk_overlay(original: np.ndarray, residual: np.ndarray, topk_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = np.stack([normalize_image(original)] * 3, axis=-1)
    scale = float(np.percentile(residual, 99))
    heat_values = np.zeros_like(residual, dtype=np.float32)
    if scale > 1e-8:
        heat_values = np.clip(residual / scale, 0.0, 1.0)
    heat_values = heat_values * topk_mask.astype(np.float32)
    heatmap = plt.get_cmap("magma")(heat_values)[..., :3]
    overlay = np.clip((1.0 - alpha) * base + alpha * heatmap, 0.0, 1.0)
    mask_edge = topk_mask.astype(bool)
    overlay[mask_edge, 0] = np.maximum(overlay[mask_edge, 0], 1.0)
    overlay[mask_edge, 1] = np.maximum(overlay[mask_edge, 1], 0.65)
    return overlay


def topk_mask_from_squared_error(squared_error: np.ndarray, percent: float) -> np.ndarray:
    flat = squared_error.reshape(-1)
    k = max(1, int(len(flat) * percent / 100.0))
    threshold = np.partition(flat, len(flat) - k)[len(flat) - k]
    return squared_error >= threshold


def load_image_tensor(path: Path, image_size: int, device: torch.device, crop_mode: str) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as image:
        tensor = build_transform(image_size, crop_mode=crop_mode)(image.convert("L")).unsqueeze(0)
    return tensor.to(device)


def append_first_unique(
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


def select_topk_explainability_rows(test_df: pd.DataFrame, score_column: str, threshold: float) -> list[tuple[str, pd.Series]]:
    selections: list[tuple[str, pd.Series]] = []
    used_paths: set[str] = set()
    normal_df = test_df[test_df["class_name"] == "NORMAL"]
    cnv_df = test_df[test_df["class_name"] == "CNV"]
    dme_df = test_df[test_df["class_name"] == "DME"]
    drusen_df = test_df[test_df["class_name"] == "DRUSEN"]

    append_first_unique(
        selections,
        used_paths,
        "NORMAL true negative",
        normal_df[normal_df[score_column] <= threshold].sort_values(score_column, ascending=True),
    )
    append_first_unique(
        selections,
        used_paths,
        "NORMAL false positive",
        normal_df[normal_df[score_column] > threshold].sort_values(score_column, ascending=False),
    )
    append_first_unique(
        selections,
        used_paths,
        "CNV true positive",
        cnv_df[cnv_df[score_column] > threshold].sort_values(score_column, ascending=False),
    )
    append_first_unique(
        selections,
        used_paths,
        "DME true positive",
        dme_df[dme_df[score_column] > threshold].sort_values(score_column, ascending=False),
    )
    append_first_unique(
        selections,
        used_paths,
        "DRUSEN true positive",
        drusen_df[drusen_df[score_column] > threshold].sort_values(score_column, ascending=False),
    )
    append_first_unique(
        selections,
        used_paths,
        "DRUSEN false negative",
        drusen_df[drusen_df[score_column] <= threshold].sort_values(score_column, ascending=False),
    )
    return selections


def select_drusen_false_negative_rows(
    test_df: pd.DataFrame,
    score_column: str,
    threshold: float,
    max_examples: int = 6,
) -> list[tuple[str, pd.Series]]:
    drusen_fn = (
        test_df[(test_df["class_name"] == "DRUSEN") & (test_df[score_column] <= threshold)]
        .sort_values(score_column, ascending=False)
        .head(max_examples)
    )
    return [(f"DRUSEN false negative #{index + 1}", row) for index, (_, row) in enumerate(drusen_fn.iterrows())]


def save_topk_explainability_grid(
    model: torch.nn.Module,
    rows: list[tuple[str, pd.Series]],
    score_column: str,
    threshold: float,
    image_size: int,
    device: torch.device,
    save_path: Path,
    crop_mode: str,
    topk_percent: float = 5.0,
) -> None:
    if not rows:
        return

    model.eval()
    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 2.6 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for row_index, (label, row) in enumerate(rows):
            image_tensor = load_image_tensor(Path(row["path"]), image_size, device, crop_mode)
            reconstruction = unpack_reconstruction(model(image_tensor))
            original = image_tensor.cpu().numpy()[0, 0]
            reconstructed = reconstruction.cpu().numpy()[0, 0]
            residual = np.abs(original - reconstructed)
            squared_error = residual**2
            topk_mask = topk_mask_from_squared_error(squared_error, topk_percent)
            overlay = build_topk_overlay(original, residual, topk_mask)
            prediction = "anomaly" if float(row[score_column]) > threshold else "normal"
            row_title = f"{label} | {row['class_name']} | {score_column}={float(row[score_column]):.5f} | pred={prediction}"

            panels = [
                ("Original", original, "gray"),
                ("Reconstruction", reconstructed, "gray"),
                ("Residual", residual, "magma"),
                (f"Top {topk_percent:g}% residual overlay", overlay, None),
            ]
            for column_index, (title, image, cmap) in enumerate(panels):
                axis = axes[row_index, column_index]
                axis.imshow(image, cmap=cmap)
                axis.set_title(f"{row_title}\n{title}", fontsize=7)
                axis.axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate alternative anomaly scores for a completed OCT experiment.")
    parser.add_argument("--run-id", default="ae_mse_l256_bs16")
    parser.add_argument("--experiments-root", default="outputs/experiments")
    parser.add_argument("--output-root", default="outputs/score_ablation")
    parser.add_argument("--checkpoint-name", default="best_autoencoder.pt")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--score-modes", nargs="*", default=BASE_SCORE_MODES)
    parser.add_argument("--patient-aggregations", nargs="*", default=["mean", "max", "top2_mean"])
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.experiments_root) / args.run_id
    metrics_dir = run_dir / "metrics"
    output_dir = Path(args.output_root) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(metrics_dir / "run_config.json")
    config["batch_size"] = args.eval_batch_size
    config["num_workers"] = args.num_workers
    set_seed(int(config["seed"]))

    device = get_device()
    data = prepare_datasets(config, device)
    model = build_model(
        model_type=config["model_type"],
        latent_dim=int(config["latent_dim"]),
        image_size=int(config["image_size"]),
        use_batch_norm=bool(config.get("use_batch_norm", False)),
    ).to(device)
    checkpoint_path = run_dir / "saved_models" / args.checkpoint_name
    model.load_state_dict(load_state_dict(checkpoint_path, device))

    base_modes = list(dict.fromkeys(args.score_modes))
    val_df = compute_score_frame(model, data["val_loader"], device, base_modes)
    test_df = compute_score_frame(model, data["test_loader"], device, base_modes)
    ensemble_modes = add_ensemble_scores(val_df, test_df, base_modes)
    all_modes = base_modes + ensemble_modes

    comparison_df, threshold_df, classwise_df = build_comparison_tables(
        val_df=val_df,
        test_df=test_df,
        score_modes=all_modes,
        threshold_percentiles=config["threshold_percentiles"],
        default_percentile=config["default_percentile"],
    )
    best_score = choose_best_score(comparison_df)
    patient_comparison_df, patient_threshold_df, patient_classwise_df, best_patient_score = build_patient_level_tables(
        val_df=val_df,
        test_df=test_df,
        score_modes=all_modes,
        threshold_percentiles=config["threshold_percentiles"],
        default_percentile=config["default_percentile"],
        aggregations=args.patient_aggregations,
    )
    bootstrap_df = build_bootstrap_ci(
        val_df=val_df,
        test_df=test_df,
        score_modes=all_modes,
        default_percentile=int(config["default_percentile"]),
        seed=int(config["seed"]),
        iterations=args.bootstrap_iterations,
        patient_aggregations=args.patient_aggregations,
    )

    val_df.to_csv(output_dir / "validation_scores.csv", index=False)
    test_df.to_csv(output_dir / "test_scores.csv", index=False)
    comparison_df.to_csv(output_dir / "score_comparison.csv", index=False)
    threshold_df.to_csv(output_dir / "threshold_comparison.csv", index=False)
    classwise_df.to_csv(output_dir / "classwise_score_summary.csv", index=False)
    patient_comparison_df.to_csv(output_dir / "patient_level_comparison.csv", index=False)
    patient_threshold_df.to_csv(output_dir / "patient_level_threshold_comparison.csv", index=False)
    patient_classwise_df.to_csv(output_dir / "patient_level_classwise_summary.csv", index=False)
    bootstrap_df.to_csv(output_dir / "bootstrap_confidence_intervals.csv", index=False)
    save_json(best_score, output_dir / "best_score.json")
    save_json(best_patient_score, output_dir / "best_patient_score.json")
    plot_score_comparison(comparison_df, output_dir / "score_comparison.png")
    plot_roc_overlay(test_df, comparison_df, output_dir / "roc_overlay.png")

    explain_mode = "topk_mse_5" if "topk_mse_5" in all_modes else str(best_score.get("score_mode", all_modes[0]))
    explain_threshold = float(np.percentile(val_df[explain_mode].to_numpy(dtype=float), int(config["default_percentile"])))
    topk_percent = 5.0
    if explain_mode.startswith("topk_mse_"):
        topk_percent = float(explain_mode.rsplit("_", 1)[-1])
    save_topk_explainability_grid(
        model=model,
        rows=select_topk_explainability_rows(test_df, explain_mode, explain_threshold),
        score_column=explain_mode,
        threshold=explain_threshold,
        image_size=int(config["image_size"]),
        device=device,
        save_path=output_dir / "topk_explainability_grid.png",
        crop_mode=config.get("crop_mode", "none"),
        topk_percent=topk_percent,
    )
    save_topk_explainability_grid(
        model=model,
        rows=select_drusen_false_negative_rows(test_df, explain_mode, explain_threshold),
        score_column=explain_mode,
        threshold=explain_threshold,
        image_size=int(config["image_size"]),
        device=device,
        save_path=output_dir / "drusen_false_negative_topk_grid.png",
        crop_mode=config.get("crop_mode", "none"),
        topk_percent=topk_percent,
    )

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "score_count": len(all_modes),
                "best_score": best_score,
                "best_patient_score": best_patient_score,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
