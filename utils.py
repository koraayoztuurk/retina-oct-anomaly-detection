from __future__ import annotations

import json
import os
import random
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

CONFIG: dict[str, Any] = {
    "data_root": os.path.join("data", "oct2017"),
    "output_dir": "outputs",
    "report_dir": "report",
    "report_template": "IEEE_Turkey_TUAC_Template_TR_2016_Final.docx",
    "seed": 42,
    "image_size": 128,
    "val_ratio": 0.2,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 40,
    "early_stopping_patience": 8,
    "latent_dim": 128,
    "num_workers": 0,
    "threshold_percentiles": [95, 97, 99],
    "default_percentile": 95,
    "train_split_name": "train",
    "test_split_name": "test",
}

CLASS_NAMES = ("NORMAL", "CNV", "DME", "DRUSEN")
SUPPORTED_EXTENSIONS = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff")


def clone_config() -> dict[str, Any]:
    return deepcopy(CONFIG)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    print(f"[INFO] PyTorch version        : {torch.__version__}")
    print(f"[INFO] CUDA available         : {torch.cuda.is_available()}")
    print(f"[INFO] CUDA device count      : {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"[INFO] Using device           : cuda")
        print(f"[INFO] GPU name              : {props.name}")
        print(f"[INFO] GPU memory (GB)       : {props.total_memory / (1024 ** 3):.2f}")
        return device

    print("[INFO] Using device           : cpu")
    return torch.device("cpu")


def prepare_output_dirs(config: dict[str, Any], clean: bool = True) -> dict[str, Path]:
    output_root = Path(config["output_dir"])
    report_root = Path(config["report_dir"])
    directories = {
        "output_root": output_root,
        "figures": output_root / "figures",
        "metrics": output_root / "metrics",
        "reconstructions": output_root / "reconstructions",
        "saved_models": output_root / "saved_models",
        "report_root": report_root,
    }

    if clean and output_root.exists():
        shutil.rmtree(output_root)

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    return directories


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved JSON            : {path}")


def save_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"[INFO] Saved text            : {path}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def ensure_dataset_root(data_root: str | Path, train_split_name: str, test_split_name: str) -> Path:
    root = Path(data_root)
    train_dir = root / train_split_name
    test_dir = root / test_split_name
    if train_dir.exists() and test_dir.exists():
        return root

    message = (
        f"Expected OCT dataset under '{root / train_split_name}' and '{root / test_split_name}'.\n"
        "Use the Kermany OCT structure: data/oct2017/{train,test}/{NORMAL,CNV,DME,DRUSEN}.\n"
        "You can also create a synthetic smoke-test dataset with:\n"
        "python scripts/create_mock_oct_dataset.py"
    )
    raise FileNotFoundError(message)


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
