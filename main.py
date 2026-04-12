from __future__ import annotations

import argparse
from pathlib import Path

from data_utils import prepare_datasets
from evaluate import (
    build_classwise_summary,
    build_summary_text,
    build_threshold_table,
    compute_reconstruction_results,
    compute_thresholds,
    evaluate_binary_metrics,
    plot_classwise_error_summary,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_test_error_distribution,
    plot_training_loss,
    plot_val_error_histogram,
    save_reconstruction_grid,
)
from model import ConvAutoencoder
from report_builder import build_report_assets
from train import train_autoencoder
from utils import clone_config, count_parameters, get_device, prepare_output_dirs, save_json, save_text, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retina OCT anomaly detection with a convolutional autoencoder.")
    parser.add_argument("--data-root", default=None, help="Dataset root. Expected structure: data/oct2017/{train,test}/{NORMAL,CNV,DME,DRUSEN}")
    parser.add_argument("--report-template", default=None, help="Path to the IEEE DOCX template.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--clean-outputs", action="store_true", help="Delete outputs/ before running.")
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    override_fields = {
        "data_root": args.data_root,
        "report_template": args.report_template,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "latent_dim": args.latent_dim,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
    }
    for key, value in override_fields.items():
        if value is not None:
            config[key] = value
    return config


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(clone_config(), args)
    set_seed(config["seed"])
    device = get_device()
    directories = prepare_output_dirs(config, clean=args.clean_outputs)

    print("\n[1/7] Preparing datasets")
    data = prepare_datasets(config, device)
    data["dataset_summary"].to_csv(directories["metrics"] / "dataset_summary.csv", index=False)

    print("\n[2/7] Building model")
    model = ConvAutoencoder(latent_dim=config["latent_dim"], image_size=config["image_size"]).to(device)
    print(model)
    print(f"[INFO] Parameter count         : {count_parameters(model):,}")

    print("\n[3/7] Training model")
    checkpoint_path = directories["saved_models"] / "best_autoencoder.pt"
    history = train_autoencoder(
        model=model,
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        device=device,
        config=config,
        checkpoint_path=checkpoint_path,
    )

    print("\n[4/7] Computing validation thresholds")
    val_results, _ = compute_reconstruction_results(model, data["val_loader"], device)
    thresholds = compute_thresholds(
        val_results["reconstruction_error"].to_numpy(),
        config["threshold_percentiles"],
    )
    default_threshold = thresholds[config["default_percentile"]]

    print("\n[5/7] Evaluating on test split")
    test_results, example_pool = compute_reconstruction_results(model, data["test_loader"], device)
    metrics = evaluate_binary_metrics(test_results, default_threshold)
    threshold_table = build_threshold_table(test_results, thresholds)
    classwise_df = build_classwise_summary(test_results)

    print("\n[6/7] Saving metrics and figures")
    val_results.to_csv(directories["metrics"] / "validation_reconstruction_errors.csv", index=False)
    test_results.to_csv(directories["metrics"] / "test_reconstruction_errors.csv", index=False)
    threshold_table.to_csv(directories["metrics"] / "threshold_comparison.csv", index=False)
    classwise_df.to_csv(directories["metrics"] / "classwise_reconstruction_summary.csv", index=False)
    save_json(config, directories["metrics"] / "run_config.json")
    save_json(metrics, directories["metrics"] / "selected_threshold_metrics.json")
    save_json(thresholds, directories["metrics"] / "thresholds.json")

    plot_training_loss(history, directories["figures"] / "training_loss.png")
    plot_val_error_histogram(val_results, thresholds, directories["figures"] / "validation_error_histogram.png")
    plot_test_error_distribution(test_results, default_threshold, directories["figures"] / "test_error_distribution.png")
    plot_roc_curve(test_results, directories["figures"] / "roc_curve.png")
    plot_confusion_matrix(metrics, directories["figures"] / "confusion_matrix.png")
    plot_classwise_error_summary(classwise_df, directories["figures"] / "classwise_error_summary.png")
    save_reconstruction_grid(example_pool, directories["reconstructions"] / "reconstruction_examples.png")

    summary_text = build_summary_text(
        config=config,
        dataset_summary=data["dataset_summary"],
        history=history,
        metrics=metrics,
        thresholds=thresholds,
        classwise_df=classwise_df,
    )
    save_text(summary_text, directories["output_root"] / "summary.txt")

    print("\n[7/7] Building report assets")
    report_context = build_report_assets(
        config=config,
        metrics=metrics,
        thresholds=thresholds,
        threshold_table=threshold_table,
        classwise_df=classwise_df,
        dataset_summary=data["dataset_summary"],
        history=history,
        output_root=directories["output_root"],
        report_root=directories["report_root"],
        template_path=Path(config["report_template"]),
    )
    save_json(report_context, directories["report_root"] / "report_context.json")

    print("\n[DONE] All outputs were generated successfully.")
    print(f"[DONE] Outputs directory       : {directories['output_root']}")
    print(f"[DONE] Report directory        : {directories['report_root']}")


if __name__ == "__main__":
    main()
