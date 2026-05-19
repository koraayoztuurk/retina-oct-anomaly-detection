from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import CLASS_NAMES, SUPPORTED_EXTENSIONS, ensure_dataset_root


PATIENT_ID_PATTERN = re.compile(r"^([A-Za-z]+)-(\d+)-")


@dataclass(frozen=True)
class OCTRecord:
    path: Path
    split_name: str
    class_name: str
    label: int
    patient_id: str


class OCTImageDataset(Dataset):
    def __init__(self, records: list[OCTRecord], transform: transforms.Compose):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("L")
            tensor = self.transform(image)

        return (
            tensor,
            int(record.label),
            record.class_name,
            record.patient_id,
            str(record.path),
        )


class SafeContentCrop:
    def __init__(
        self,
        high_threshold: float = 0.98,
        signal_threshold: float = 0.45,
        min_row_std: float = 0.03,
        max_white_fraction: float = 0.35,
        margin_ratio: float = 0.10,
        min_crop_fraction: float = 0.50,
    ):
        self.high_threshold = high_threshold
        self.signal_threshold = signal_threshold
        self.min_row_std = min_row_std
        self.max_white_fraction = max_white_fraction
        self.margin_ratio = margin_ratio
        self.min_crop_fraction = min_crop_fraction

    def __call__(self, image: Image.Image) -> Image.Image:
        grayscale = image.convert("L")
        array = np.asarray(grayscale, dtype=np.float32) / 255.0
        height, width = array.shape

        # OCT pathology can sit close to the sides, so this crop only removes
        # vertical empty bands and keeps the full horizontal field of view.
        x_start = int(width * 0.08)
        x_end = max(x_start + 1, int(width * 0.92))
        center = array[:, x_start:x_end]
        row_signal = np.percentile(center, 98, axis=1)
        row_std = center.std(axis=1)
        white_fraction = (center > self.high_threshold).mean(axis=1)
        active_rows = np.where(
            (row_signal >= self.signal_threshold)
            & (row_std >= self.min_row_std)
            & (white_fraction <= self.max_white_fraction)
        )[0]
        if len(active_rows) == 0:
            return grayscale

        y_min, y_max = int(active_rows[0]), int(active_rows[-1])
        y_margin = max(4, int(height * self.margin_ratio))
        y_min = max(0, y_min - y_margin)
        y_max = min(height - 1, y_max + y_margin)

        crop_height = y_max - y_min + 1
        if crop_height < height * self.min_crop_fraction:
            return grayscale

        if crop_height >= height * 0.98:
            return grayscale

        return grayscale.crop((0, y_min, width, y_max + 1))


class ConservativeBorderCrop:
    def __init__(
        self,
        dark_threshold: float = 0.23,
        bright_threshold: float = 0.97,
        max_dark_std: float = 0.08,
        min_blank_rows_to_crop: int = 18,
        keep_blank_rows: int = 16,
        max_crop_fraction: float = 0.08,
        center_width_fraction: float = 0.88,
    ):
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.max_dark_std = max_dark_std
        self.min_blank_rows_to_crop = min_blank_rows_to_crop
        self.keep_blank_rows = keep_blank_rows
        self.max_crop_fraction = max_crop_fraction
        self.center_width_fraction = center_width_fraction

    def _blank_row_mask(self, array: np.ndarray) -> np.ndarray:
        height, width = array.shape
        crop_margin = int(width * (1.0 - self.center_width_fraction) / 2.0)
        center = array[:, crop_margin : width - crop_margin] if crop_margin > 0 else array

        row_p90 = np.percentile(center, 90, axis=1)
        row_mean = center.mean(axis=1)
        row_std = center.std(axis=1)
        white_fraction = (center >= self.bright_threshold).mean(axis=1)

        dark_blank = (row_p90 <= self.dark_threshold) & (row_std <= self.max_dark_std)
        bright_blank = (white_fraction >= 0.60) | (row_mean >= self.bright_threshold)
        return dark_blank | bright_blank

    def _edge_crop_amount(self, blank_mask: np.ndarray, from_top: bool, image_height: int) -> int:
        indices = range(image_height) if from_top else range(image_height - 1, -1, -1)
        blank_count = 0
        for index in indices:
            if not blank_mask[index]:
                break
            blank_count += 1

        if blank_count < self.min_blank_rows_to_crop:
            return 0

        max_crop = max(1, int(image_height * self.max_crop_fraction))
        return min(max(0, blank_count - self.keep_blank_rows), max_crop)

    def __call__(self, image: Image.Image) -> Image.Image:
        grayscale = image.convert("L")
        array = np.asarray(grayscale, dtype=np.float32) / 255.0
        height, width = array.shape
        blank_mask = self._blank_row_mask(array)

        top_crop = self._edge_crop_amount(blank_mask, from_top=True, image_height=height)
        bottom_crop = self._edge_crop_amount(blank_mask, from_top=False, image_height=height)
        if top_crop == 0 and bottom_crop == 0:
            return grayscale

        y_min = top_crop
        y_max = height - bottom_crop
        if y_max - y_min < height * 0.80:
            return grayscale

        return grayscale.crop((0, y_min, width, y_max))


class RetinaMarginCrop:
    def __init__(
        self,
        signal_threshold: float = 0.42,
        signal_percentile: int = 97,
        min_row_std: float = 0.035,
        max_white_fraction: float = 0.45,
        center_width_fraction: float = 0.88,
        guard_margin_ratio: float = 0.14,
        min_guard_px: int = 60,
        min_crop_px: int = 12,
        max_edge_crop_fraction: float = 0.42,
        min_remaining_fraction: float = 0.60,
        smooth_window: int = 7,
        min_active_in_window: int = 3,
    ):
        self.signal_threshold = signal_threshold
        self.signal_percentile = signal_percentile
        self.min_row_std = min_row_std
        self.max_white_fraction = max_white_fraction
        self.center_width_fraction = center_width_fraction
        self.guard_margin_ratio = guard_margin_ratio
        self.min_guard_px = min_guard_px
        self.min_crop_px = min_crop_px
        self.max_edge_crop_fraction = max_edge_crop_fraction
        self.min_remaining_fraction = min_remaining_fraction
        self.smooth_window = smooth_window
        self.min_active_in_window = min_active_in_window

    def _retina_row_mask(self, array: np.ndarray) -> np.ndarray:
        height, width = array.shape
        crop_margin = int(width * (1.0 - self.center_width_fraction) / 2.0)
        center = array[:, crop_margin : width - crop_margin] if crop_margin > 0 else array

        row_signal = np.percentile(center, self.signal_percentile, axis=1)
        row_std = center.std(axis=1)
        white_fraction = (center >= 0.98).mean(axis=1)
        raw_mask = (
            (row_signal >= self.signal_threshold)
            & (row_std >= self.min_row_std)
            & (white_fraction <= self.max_white_fraction)
        )

        if self.smooth_window <= 1:
            return raw_mask
        kernel = np.ones(self.smooth_window, dtype=np.int32)
        active_counts = np.convolve(raw_mask.astype(np.int32), kernel, mode="same")
        return active_counts >= self.min_active_in_window

    def __call__(self, image: Image.Image) -> Image.Image:
        grayscale = image.convert("L")
        array = np.asarray(grayscale, dtype=np.float32) / 255.0
        height, width = array.shape

        retina_rows = np.where(self._retina_row_mask(array))[0]
        if len(retina_rows) == 0:
            return grayscale

        first_retina_row = int(retina_rows[0])
        last_retina_row = int(retina_rows[-1])
        guard_margin = max(self.min_guard_px, int(height * self.guard_margin_ratio))
        max_edge_crop = max(self.min_crop_px, int(height * self.max_edge_crop_fraction))

        top_crop = min(max(0, first_retina_row - guard_margin), max_edge_crop)
        bottom_crop = min(max(0, height - last_retina_row - 1 - guard_margin), max_edge_crop)
        if top_crop < self.min_crop_px:
            top_crop = 0
        if bottom_crop < self.min_crop_px:
            bottom_crop = 0

        y_min = top_crop
        y_max = height - bottom_crop
        if y_max - y_min < height * self.min_remaining_fraction:
            return grayscale
        if top_crop == 0 and bottom_crop == 0:
            return grayscale

        return grayscale.crop((0, y_min, width, y_max))


def parse_patient_id(path: Path) -> str:
    match = PATIENT_ID_PATTERN.match(path.stem)
    if match:
        return f"{match.group(1).upper()}-{match.group(2)}"
    parts = path.stem.split("-")
    class_prefix = path.parent.name.upper() if path.parent.name else "UNKNOWN"
    if parts and parts[0]:
        return f"{class_prefix}-{parts[0]}"
    return f"{class_prefix}-{path.stem}"


def collect_image_paths(directory: Path) -> list[Path]:
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def build_records(paths: Iterable[Path], split_name: str, class_name: str) -> list[OCTRecord]:
    label = 0 if class_name == "NORMAL" else 1
    return [
        OCTRecord(
            path=path,
            split_name=split_name,
            class_name=class_name,
            label=label,
            patient_id=parse_patient_id(path),
        )
        for path in paths
    ]


def build_transform(image_size: int, crop_mode: str = "none") -> transforms.Compose:
    transform_steps = []
    if crop_mode == "content":
        transform_steps.append(SafeContentCrop())
    elif crop_mode == "border":
        transform_steps.append(ConservativeBorderCrop())
    elif crop_mode == "retina_margin":
        transform_steps.append(RetinaMarginCrop())
    elif crop_mode != "none":
        raise ValueError(f"Unsupported crop_mode: {crop_mode}")

    transform_steps.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return transforms.Compose(transform_steps)


def records_to_frame(records: list[OCTRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["split_name", "class_name", "label", "patient_id", "path"])

    return pd.DataFrame(
        [
            {
                "split_name": record.split_name,
                "class_name": record.class_name,
                "label": record.label,
                "patient_id": record.patient_id,
                "path": str(record.path),
            }
            for record in records
        ]
    )


def summarize_records(records: list[OCTRecord]) -> pd.DataFrame:
    frame = records_to_frame(records)
    if frame.empty:
        return frame

    summary = (
        frame.groupby(["split_name", "class_name"], as_index=False)
        .agg(image_count=("path", "count"), patient_count=("patient_id", "nunique"))
        .sort_values(["split_name", "class_name"])
        .reset_index(drop=True)
    )
    return summary


def split_normal_train_val(
    normal_train_paths: list[Path],
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    patient_to_paths: dict[str, list[Path]] = {}
    for path in normal_train_paths:
        patient_id = parse_patient_id(path)
        patient_to_paths.setdefault(patient_id, []).append(path)

    patient_ids = sorted(patient_to_paths)
    train_patients, val_patients = train_test_split(
        patient_ids,
        test_size=val_ratio,
        random_state=seed,
    )

    train_paths = [path for patient_id in train_patients for path in patient_to_paths[patient_id]]
    val_paths = [path for patient_id in val_patients for path in patient_to_paths[patient_id]]
    return sorted(train_paths), sorted(val_paths)


def create_loader(
    records: list[OCTRecord],
    image_size: int,
    crop_mode: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = OCTImageDataset(records, build_transform(image_size, crop_mode=crop_mode))
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(
        **loader_kwargs,
    )


def prepare_datasets(config: dict, device: torch.device) -> dict:
    crop_mode = config.get("crop_mode", "none")
    data_root = ensure_dataset_root(
        config["data_root"],
        config["train_split_name"],
        config["test_split_name"],
    )
    train_root = data_root / config["train_split_name"]
    test_root = data_root / config["test_split_name"]

    normal_train_paths = collect_image_paths(train_root / "NORMAL")
    if not normal_train_paths:
        raise FileNotFoundError(f"No NORMAL images found under {train_root / 'NORMAL'}")

    train_paths, val_paths = split_normal_train_val(
        normal_train_paths=normal_train_paths,
        val_ratio=config["val_ratio"],
        seed=config["seed"],
    )

    train_records = build_records(train_paths, "train", "NORMAL")
    val_records = build_records(val_paths, "val", "NORMAL")
    test_records: list[OCTRecord] = []
    for class_name in CLASS_NAMES:
        class_paths = collect_image_paths(test_root / class_name)
        test_records.extend(build_records(class_paths, "test", class_name))

    if not test_records:
        raise FileNotFoundError(f"No test images found under {test_root}")

    pin_memory = device.type == "cuda"
    train_loader = create_loader(
        records=train_records,
        image_size=config["image_size"],
        crop_mode=crop_mode,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = create_loader(
        records=val_records,
        image_size=config["image_size"],
        crop_mode=crop_mode,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )
    test_loader = create_loader(
        records=test_records,
        image_size=config["image_size"],
        crop_mode=crop_mode,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )

    train_stats = summarize_records(train_records)
    val_stats = summarize_records(val_records)
    test_stats = summarize_records(test_records)
    dataset_summary = pd.concat([train_stats, val_stats, test_stats], ignore_index=True)
    train_frame = records_to_frame(train_records)
    val_frame = records_to_frame(val_records)
    test_frame = records_to_frame(test_records)
    overlap_patients = sorted(set(train_frame["patient_id"]) & set(val_frame["patient_id"]))
    dataset_checks = {
        "train_class_names": sorted(train_frame["class_name"].unique().tolist()),
        "train_pathology_count": int(train_frame["label"].sum()),
        "train_val_patient_overlap_count": len(overlap_patients),
        "train_val_patient_overlap_examples": overlap_patients[:5],
        "crop_mode": crop_mode,
    }

    print("[INFO] Dataset summary")
    print(dataset_summary.to_string(index=False))
    print("[INFO] Dataset checks")
    print(f"[INFO] Train classes           : {', '.join(dataset_checks['train_class_names'])}")
    print(f"[INFO] Train pathology count   : {dataset_checks['train_pathology_count']}")
    print(f"[INFO] Train/val patient overlap: {dataset_checks['train_val_patient_overlap_count']}")
    print(f"[INFO] Crop mode               : {dataset_checks['crop_mode']}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "dataset_summary": dataset_summary,
        "dataset_checks": dataset_checks,
        "train_frame": train_frame,
        "val_frame": val_frame,
        "test_frame": test_frame,
    }
