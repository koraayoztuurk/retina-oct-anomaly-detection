from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from compare_experiments import aggregate_experiments
from utils import validate_clean_output_root


CLI_FIELDS = {
    "data_root": "--data-root",
    "epochs": "--epochs",
    "early_stopping_patience": "--early-stopping-patience",
    "batch_size": "--batch-size",
    "image_size": "--image-size",
    "crop_mode": "--crop-mode",
    "latent_dim": "--latent-dim",
    "learning_rate": "--learning-rate",
    "lr_scheduler": "--lr-scheduler",
    "lr_scheduler_factor": "--lr-scheduler-factor",
    "lr_scheduler_patience": "--lr-scheduler-patience",
    "min_learning_rate": "--min-learning-rate",
    "num_workers": "--num-workers",
    "model_type": "--model-type",
    "loss_type": "--loss-type",
    "beta": "--beta",
    "default_percentile": "--default-percentile",
    "threshold_percentiles": "--threshold-percentiles",
}


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_experiment(defaults: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    merged = defaults.copy()
    merged.update(experiment)
    return merged


def build_command(experiment: dict[str, Any], clean_outputs: bool) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "main.py",
        "--run-id",
        str(experiment["run_id"]),
        "--skip-report",
    ]
    if clean_outputs:
        command.append("--clean-outputs")

    for key, flag in CLI_FIELDS.items():
        if key in experiment and experiment[key] is not None:
            value = experiment[key]
            command.append(flag)
            if isinstance(value, list):
                command.extend(str(item) for item in value)
            else:
                command.append(str(value))
    return command


def clean_run_dir(run_dir: Path, experiments_root: Path) -> None:
    resolved_root = experiments_root.resolve()
    resolved_run_dir = run_dir.resolve()
    if resolved_run_dir == resolved_root:
        raise ValueError(f"Refusing to delete experiments root directly: {run_dir}")
    if resolved_root not in [resolved_run_dir, *resolved_run_dir.parents]:
        raise ValueError(f"Refusing to delete run directory outside experiments root: {run_dir}")
    if run_dir.exists():
        shutil.rmtree(validate_clean_output_root(run_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final Retina OCT experiment matrix.")
    parser.add_argument("--config", default="configs/final_experiments.json")
    parser.add_argument("--experiments-root", default="outputs/experiments")
    parser.add_argument("--comparison-root", default="outputs/comparison")
    parser.add_argument("--only", nargs="*", default=None, help="Optional list of run_id values to execute.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with existing selected metrics.")
    parser.add_argument("--clean-outputs", action="store_true", help="Clean each run output folder before execution.")
    parser.add_argument("--write-logs", action="store_true", help="Write each run output to outputs/experiments/<run_id>/run.log.")
    parser.add_argument("--aggregate-config-only", action="store_true", help="Aggregate only run_id values listed in this config.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(Path(args.config))
    defaults = config.get("defaults", {})
    experiments = [merge_experiment(defaults, item) for item in config.get("experiments", [])]
    if args.only:
        allowed = set(args.only)
        experiments = [item for item in experiments if item["run_id"] in allowed]

    for experiment in experiments:
        run_id = str(experiment["run_id"])
        metrics_path = Path(args.experiments_root) / run_id / "metrics" / "selected_threshold_metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"[SKIP] {run_id}: metrics already exist")
            continue

        run_dir = Path(args.experiments_root) / run_id
        if args.write_logs and args.clean_outputs:
            clean_run_dir(run_dir, Path(args.experiments_root))

        command = build_command(experiment, clean_outputs=args.clean_outputs and not args.write_logs)
        print("[RUN] " + " ".join(command))
        if not args.dry_run:
            if args.write_logs:
                log_path = run_dir / "run.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as log_file:
                    subprocess.run(command, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            else:
                subprocess.run(command, check=True)

    if not args.dry_run:
        include_run_ids = {str(item["run_id"]) for item in experiments} if args.aggregate_config_only else None
        summary = aggregate_experiments(
            Path(args.experiments_root),
            Path(args.comparison_root),
            include_run_ids=include_run_ids,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
