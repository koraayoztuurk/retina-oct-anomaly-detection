from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_score_tables(score_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_frames: list[pd.DataFrame] = []
    classwise_frames: list[pd.DataFrame] = []
    patient_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []

    for run_dir in sorted(path for path in score_root.iterdir() if path.is_dir()):
        comparison_path = run_dir / "score_comparison.csv"
        classwise_path = run_dir / "classwise_score_summary.csv"
        patient_path = run_dir / "patient_level_comparison.csv"
        bootstrap_path = run_dir / "bootstrap_confidence_intervals.csv"

        if comparison_path.exists():
            frame = pd.read_csv(comparison_path)
            frame.insert(0, "run_id", run_dir.name)
            comparison_frames.append(frame)

        if classwise_path.exists():
            frame = pd.read_csv(classwise_path)
            frame.insert(0, "run_id", run_dir.name)
            classwise_frames.append(frame)

        if patient_path.exists():
            frame = pd.read_csv(patient_path)
            frame.insert(0, "run_id", run_dir.name)
            patient_frames.append(frame)

        if bootstrap_path.exists():
            frame = pd.read_csv(bootstrap_path)
            frame.insert(0, "run_id", run_dir.name)
            bootstrap_frames.append(frame)

    comparison_df = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    classwise_df = pd.concat(classwise_frames, ignore_index=True) if classwise_frames else pd.DataFrame()
    patient_df = pd.concat(patient_frames, ignore_index=True) if patient_frames else pd.DataFrame()
    bootstrap_df = pd.concat(bootstrap_frames, ignore_index=True) if bootstrap_frames else pd.DataFrame()
    return comparison_df, classwise_df, patient_df, bootstrap_df


def choose_best_per_run(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        return comparison_df

    best_rows = []
    for run_id, run_frame in comparison_df.groupby("run_id", sort=True):
        sorted_frame = run_frame.sort_values(["auroc", "f1", "recall"], ascending=False).reset_index(drop=True)
        best = sorted_frame.iloc[0]
        tied = run_frame[run_frame["auroc"] >= best["auroc"] - 0.005].copy()
        if len(tied) > 1:
            best = tied.sort_values(["f1", "recall"], ascending=False).iloc[0]
        tied_f1 = run_frame[
            (run_frame["auroc"] >= best["auroc"] - 0.005)
            & (run_frame["f1"] >= best["f1"] - 0.01)
        ].copy()
        if len(tied_f1) > 1:
            best = tied_f1.sort_values(["recall", "f1", "auroc"], ascending=False).iloc[0]
        best_rows.append(best)

    return pd.DataFrame(best_rows).sort_values(["auroc", "f1", "recall"], ascending=False)


def choose_best_overall(best_df: pd.DataFrame) -> dict:
    if best_df.empty:
        return {}

    sorted_frame = best_df.sort_values(["auroc", "f1", "recall"], ascending=False).reset_index(drop=True)
    best = sorted_frame.iloc[0]
    tied = best_df[best_df["auroc"] >= best["auroc"] - 0.005].copy()
    if len(tied) > 1:
        best = tied.sort_values(["f1", "recall"], ascending=False).iloc[0]
    tied_f1 = best_df[
        (best_df["auroc"] >= best["auroc"] - 0.005)
        & (best_df["f1"] >= best["f1"] - 0.01)
    ].copy()
    if len(tied_f1) > 1:
        best = tied_f1.sort_values(["recall", "f1", "auroc"], ascending=False).iloc[0]
    return best.to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed anomaly score ablation outputs.")
    parser.add_argument("--score-root", default="outputs/score_ablation")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_root = Path(args.score_root)
    if not score_root.exists():
        raise FileNotFoundError(f"Score ablation root not found: {score_root}")

    output_dir = Path(args.output_dir) if args.output_dir else score_root
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df, classwise_df, patient_df, bootstrap_df = read_score_tables(score_root)
    best_df = choose_best_per_run(comparison_df)
    best_overall = choose_best_overall(best_df)
    best_patient_df = choose_best_per_run(patient_df) if not patient_df.empty else pd.DataFrame()
    best_patient_overall = choose_best_overall(best_patient_df)

    comparison_path = output_dir / "score_ablation_summary.csv"
    best_path = output_dir / "best_score_by_run.csv"
    best_overall_path = output_dir / "best_score_overall.json"
    classwise_path = output_dir / "score_ablation_classwise_summary.csv"
    patient_path = output_dir / "patient_level_score_summary.csv"
    best_patient_path = output_dir / "best_patient_score_by_run.csv"
    best_patient_overall_path = output_dir / "best_patient_score_overall.json"
    bootstrap_path = output_dir / "bootstrap_confidence_intervals_summary.csv"

    comparison_df.to_csv(comparison_path, index=False)
    best_df.to_csv(best_path, index=False)
    classwise_df.to_csv(classwise_path, index=False)
    patient_df.to_csv(patient_path, index=False)
    best_patient_df.to_csv(best_patient_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)
    with best_overall_path.open("w", encoding="utf-8") as handle:
        json.dump(best_overall, handle, indent=2, ensure_ascii=False)
    with best_patient_overall_path.open("w", encoding="utf-8") as handle:
        json.dump(best_patient_overall, handle, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved score summary      : {comparison_path}")
    print(f"[INFO] Saved best-by-run       : {best_path}")
    print(f"[INFO] Saved best overall      : {best_overall_path}")
    print(f"[INFO] Saved classwise summary : {classwise_path}")
    print(f"[INFO] Saved patient summary    : {patient_path}")
    print(f"[INFO] Saved patient best       : {best_patient_path}")
    print(f"[INFO] Saved bootstrap summary  : {bootstrap_path}")
    if not best_df.empty:
        print(best_df[["run_id", "score_mode", "auroc", "f1", "recall", "precision", "fpr"]].to_string(index=False))
    if best_overall:
        print("[INFO] Best overall candidate")
        print(
            pd.DataFrame([best_overall])[
                ["run_id", "score_mode", "auroc", "f1", "recall", "precision", "fpr"]
            ].to_string(index=False)
        )
    if best_patient_overall:
        print("[INFO] Best patient-level candidate")
        print(
            pd.DataFrame([best_patient_overall])[
                ["run_id", "patient_aggregation", "score_mode", "auroc", "f1", "recall", "precision", "fpr"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
