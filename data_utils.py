from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import CLASS_NAMES, SUPPORTED_EXTENSIONS, ensure_dataset_root


PATIENT_ID_PATTERN = re.compile(r"^[A-Za-z]+-(\d+)-")


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


def parse_patient_id(path: Path) -> str:
    match = PATIENT_ID_PATTERN.match(path.stem)
    if match:
        return match.group(1)
    parts = path.stem.split("-")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return path.stem


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


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


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
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = OCTImageDataset(records, build_transform(image_size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def prepare_datasets(config: dict, device: torch.device) -> dict:
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
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = create_loader(
        records=val_records,
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )
    test_loader = create_loader(
        records=test_records,
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )

    train_stats = summarize_records(train_records)
    val_stats = summarize_records(val_records)
    test_stats = summarize_records(test_records)
    dataset_summary = pd.concat([train_stats, val_stats, test_stats], ignore_index=True)

    print("[INFO] Dataset summary")
    print(dataset_summary.to_string(index=False))

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "dataset_summary": dataset_summary,
        "train_frame": records_to_frame(train_records),
        "val_frame": records_to_frame(val_records),
        "test_frame": records_to_frame(test_records),
    }
