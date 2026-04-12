from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = ("NORMAL", "CNV", "DME", "DRUSEN")


def draw_base_layer_pattern(size: int = 128, noise_scale: float = 0.03) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.float32)
    x = np.linspace(0.0, 1.0, size)
    layer_1 = (0.28 + 0.02 * np.sin(2 * math.pi * x * 1.2)) * size
    layer_2 = (0.48 + 0.015 * np.sin(2 * math.pi * x * 1.6 + 0.5)) * size
    layer_3 = (0.68 + 0.01 * np.sin(2 * math.pi * x * 1.8 + 1.0)) * size

    for column in range(size):
        for layer in [layer_1, layer_2, layer_3]:
            center = int(layer[column])
            image[max(center - 1, 0): min(center + 2, size), column] = 0.65

    image += np.random.normal(0.1, noise_scale, size=(size, size)).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def add_cnv_artifact(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    rr, cc = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = ((rr - 48) ** 2) / 100 + ((cc - 64) ** 2) / 500 <= 1
    image[mask] = np.clip(image[mask] + 0.45, 0.0, 1.0)
    return image


def add_dme_artifact(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    for center_x in [42, 64, 88]:
        rr, cc = np.ogrid[: image.shape[0], : image.shape[1]]
        mask = ((rr - 74) ** 2) / 60 + ((cc - center_x) ** 2) / 60 <= 1
        image[mask] = np.clip(image[mask] + 0.55, 0.0, 1.0)
    return image


def add_drusen_artifact(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    for center_x in [34, 58, 84, 102]:
        rr, cc = np.ogrid[: image.shape[0], : image.shape[1]]
        mask = ((rr - 88) ** 2) / 25 + ((cc - center_x) ** 2) / 45 <= 1
        image[mask] = np.clip(image[mask] + 0.35, 0.0, 1.0)
    return image


def artifact_for_class(class_name: str, base_image: np.ndarray) -> np.ndarray:
    if class_name == "CNV":
        return add_cnv_artifact(base_image)
    if class_name == "DME":
        return add_dme_artifact(base_image)
    if class_name == "DRUSEN":
        return add_drusen_artifact(base_image)
    return base_image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image * 255).astype(np.uint8)).save(path)


def generate_split(root: Path, split_name: str, patients_per_class: int, images_per_patient: int) -> None:
    for class_name in CLASSES:
        for patient_index in range(patients_per_class):
            patient_id = 1000 + patient_index
            for image_index in range(images_per_patient):
                base_image = draw_base_layer_pattern()
                image = artifact_for_class(class_name, base_image)
                file_name = f"{class_name}-{patient_id:04d}-{image_index + 1}.jpeg"
                save_image(root / split_name / class_name / file_name, image)


def clear_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def main() -> None:
    root = Path("data/mock_oct2017")
    clear_tree(root)
    generate_split(root, "train", patients_per_class=10, images_per_patient=3)
    generate_split(root, "test", patients_per_class=5, images_per_patient=2)
    print(f"Mock OCT dataset created at: {root}")


if __name__ == "__main__":
    main()
