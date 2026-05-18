from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from data_utils import ConservativeBorderCrop, RetinaMarginCrop, SafeContentCrop, collect_image_paths
from utils import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a visual preview for OCT preprocessing choices.")
    parser.add_argument("--data-root", default="data/oct2017")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output", default="outputs/preprocessing/content_crop_preview.png")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--crop-mode", choices=["content", "border", "retina_margin"], default="content")
    return parser.parse_args()


def resize_for_preview(image: Image.Image, size: int = 180) -> Image.Image:
    return image.convert("L").resize((size, size))


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root) / args.split
    cropper_by_mode = {
        "content": SafeContentCrop,
        "border": ConservativeBorderCrop,
        "retina_margin": RetinaMarginCrop,
    }
    cropper = cropper_by_mode[args.crop_mode]()
    rows: list[tuple[str, Path]] = []
    for class_name in CLASS_NAMES:
        rows.extend((class_name, path) for path in collect_image_paths(data_root / class_name)[: args.samples_per_class])

    if not rows:
        raise FileNotFoundError(f"No images found under {data_root}")

    fig, axes = plt.subplots(len(rows), 2, figsize=(7, 2.4 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(1, -1)

    for row_index, (class_name, path) in enumerate(rows):
        with Image.open(path) as image:
            original = image.convert("L")
            cropped = cropper(original)
            original_preview = resize_for_preview(original)
            cropped_preview = resize_for_preview(cropped)

        axes[row_index, 0].imshow(original_preview, cmap="gray")
        axes[row_index, 0].set_title(f"{class_name} original\n{original.size[0]}x{original.size[1]}", fontsize=8)
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(cropped_preview, cmap="gray")
        axes[row_index, 1].set_title(f"{args.crop_mode} crop\n{cropped.size[0]}x{cropped.size[1]}", fontsize=8)
        axes[row_index, 1].axis("off")

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
