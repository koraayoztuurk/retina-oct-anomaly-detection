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
    save_drusen_false_negative_artifacts,
    save_ranked_reconstruction_grid,
    save_reconstruction_grid,
    save_residual_heatmap_overlay_grid,
)
from model import build_model
from train import train_autoencoder
from utils import clone_config, count_parameters, get_device, prepare_output_dirs, save_json, save_text, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retina OCT anomaly detection with a convolutional autoencoder.")
    parser.add_argument("--data-root", default=None, help="Dataset root. Expected structure: data/oct2017/{train,test}/{NORMAL,CNV,DME,DRUSEN}")
    parser.add_argument("--report-template", default=None, help="Path to the IEEE DOCX template.")
    parser.add_argument("--run-id", default=None, help="Experiment id. Outputs are stored under outputs/experiments/<run-id>.")
    parser.add_argument("--model-type", choices=["ae", "vae"], default=None)
    parser.add_argument("--use-batch-norm", action="store_true", default=None)
    parser.add_argument("--loss-type", choices=["mse", "l1", "mse_ssim", "vae_mse_kl"], default=None)
    parser.add_argument("--beta", type=float, default=None, help="KL weight for VAE runs.")
    parser.add_argument("--output-root", default=None, help="Direct output root override for this run.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--crop-mode", choices=["none", "content", "border", "retina_margin"], default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default=None)
    parser.add_argument("--lr-scheduler-factor", type=float, default=None)
    parser.add_argument("--lr-scheduler-patience", type=int, default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--default-percentile", type=int, default=None)
    parser.add_argument("--threshold-percentiles", type=int, nargs="+", default=None)
    parser.add_argument("--clean-outputs", action="store_true", help="Delete this run output folder before running.")
    parser.add_argument("--skip-report", action="store_true", help="Skip ara-report asset generation for experiment runs.")
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    override_fields = {
        "data_root": args.data_root,
        "output_root": args.output_root,
        "report_template": args.report_template,
        "run_id": args.run_id,
        "model_type": args.model_type,
        "use_batch_norm": args.use_batch_norm,
        "loss_type": args.loss_type,
        "beta": args.beta,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "crop_mode": args.crop_mode,
        "latent_dim": args.latent_dim,
        "learning_rate": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "min_learning_rate": args.min_learning_rate,
        "num_workers": args.num_workers,
        "default_percentile": args.default_percentile,
        "threshold_percentiles": args.threshold_percentiles,
    }
    for key, value in override_fields.items():
        if value is not None:
            config[key] = value
    return config


def validate_config(config: dict) -> None:
    if config["epochs"] < 1:
        raise ValueError("epochs must be at least 1.")
    if config["batch_size"] < 1:
        raise ValueError("batch_size must be at least 1.")
    if config["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative.")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive.")
    if config["lr_scheduler"] not in {"none", "plateau"}:
        raise ValueError("lr_scheduler must be one of: none, plateau.")
    if not 0 < float(config["lr_scheduler_factor"]) < 1:
        raise ValueError("lr_scheduler_factor must be between 0 and 1.")
    if int(config["lr_scheduler_patience"]) < 1:
        raise ValueError("lr_scheduler_patience must be at least 1.")
    if float(config["min_learning_rate"]) <= 0:
        raise ValueError("min_learning_rate must be positive.")
    if not config["threshold_percentiles"]:
        raise ValueError("threshold_percentiles cannot be empty.")
    if config["default_percentile"] not in config["threshold_percentiles"]:
        raise ValueError("default_percentile must be included in threshold_percentiles.")
    if config["loss_type"] == "vae_mse_kl" and config["model_type"] != "vae":
        raise ValueError("loss_type=vae_mse_kl requires --model-type vae.")
    if config["model_type"] == "vae" and config["loss_type"] != "vae_mse_kl":
        raise ValueError("model_type=vae currently expects --loss-type vae_mse_kl.")
    if not isinstance(config["use_batch_norm"], bool):
        raise ValueError("use_batch_norm must be a boolean.")


def score_type_for_config(config: dict) -> str:
    if config["loss_type"] == "vae_mse_kl":
        return "mse"
    return config["loss_type"]


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(clone_config(), args)
    validate_config(config)
    set_seed(config["seed"])
    device = get_device()
    directories = prepare_output_dirs(config, clean=args.clean_outputs)

    print("\n[1/7] Preparing datasets")
    data = prepare_datasets(config, device)
    data["dataset_summary"].to_csv(directories["metrics"] / "dataset_summary.csv", index=False)

    print("\n[2/7] Building model")
    model = build_model(
        model_type=config["model_type"],
        latent_dim=config["latent_dim"],
        image_size=config["image_size"],
        use_batch_norm=bool(config.get("use_batch_norm", False)),
    ).to(device)
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
    score_type = score_type_for_config(config)
    val_results, _ = compute_reconstruction_results(model, data["val_loader"], device, score_type=score_type)
    thresholds = compute_thresholds(
        val_results["reconstruction_error"].to_numpy(),
        config["threshold_percentiles"],
    )
    default_threshold = thresholds[config["default_percentile"]]

    print("\n[5/7] Evaluating on test split")
    test_results, example_pool = compute_reconstruction_results(model, data["test_loader"], device, score_type=score_type)
    metrics = evaluate_binary_metrics(test_results, default_threshold)
    threshold_table = build_threshold_table(test_results, thresholds)
    classwise_df = build_classwise_summary(test_results, threshold=default_threshold)
    metrics.update(
        {
            "run_id": config.get("run_id"),
            "model_type": config["model_type"],
            "loss_type": config["loss_type"],
            "score_type": score_type,
            "default_percentile": config["default_percentile"],
        }
    )

    print("\n[6/7] Saving metrics and figures")
    val_results.to_csv(directories["metrics"] / "validation_reconstruction_errors.csv", index=False)
    test_results.to_csv(directories["metrics"] / "test_reconstruction_errors.csv", index=False)
    threshold_table.to_csv(directories["metrics"] / "threshold_comparison.csv", index=False)
    classwise_df.to_csv(directories["metrics"] / "classwise_reconstruction_summary.csv", index=False)
    save_json(config, directories["metrics"] / "run_config.json")
    save_json(history, directories["metrics"] / "training_history.json")
    save_json(data["dataset_checks"], directories["metrics"] / "dataset_checks.json")
    save_json(metrics, directories["metrics"] / "selected_threshold_metrics.json")
    save_json(thresholds, directories["metrics"] / "thresholds.json")

    plot_training_loss(history, directories["figures"] / "training_loss.png")
    plot_val_error_histogram(val_results, thresholds, directories["figures"] / "validation_error_histogram.png")
    plot_test_error_distribution(test_results, default_threshold, directories["figures"] / "test_error_distribution.png")
    plot_roc_curve(test_results, directories["figures"] / "roc_curve.png")
    plot_confusion_matrix(metrics, directories["figures"] / "confusion_matrix.png")
    plot_classwise_error_summary(classwise_df, directories["figures"] / "classwise_error_summary.png")
    save_reconstruction_grid(example_pool, directories["reconstructions"] / "reconstruction_examples.png")
    save_ranked_reconstruction_grid(
        model=model,
        results_df=test_results,
        threshold=default_threshold,
        image_size=config["image_size"],
        device=device,
        save_path=directories["reconstructions"] / "best_worst_reconstruction_examples.png",
        crop_mode=config["crop_mode"],
    )
    save_residual_heatmap_overlay_grid(
        model=model,
        results_df=test_results,
        threshold=default_threshold,
        image_size=config["image_size"],
        device=device,
        save_path=directories["reconstructions"] / "residual_heatmap_overlay_examples.png",
        crop_mode=config["crop_mode"],
    )
    save_drusen_false_negative_artifacts(
        model=model,
        results_df=test_results,
        threshold=default_threshold,
        image_size=config["image_size"],
        device=device,
        figure_path=directories["reconstructions"] / "drusen_false_negative_examples.png",
        csv_path=directories["metrics"] / "drusen_false_negative_examples.csv",
        crop_mode=config["crop_mode"],
    )

    summary_text = build_summary_text(
        config=config,
        dataset_summary=data["dataset_summary"],
        history=history,
        metrics=metrics,
        thresholds=thresholds,
        classwise_df=classwise_df,
    )
    save_text(summary_text, directories["output_root"] / "summary.txt")

    if args.skip_report:
        print("\n[7/7] Skipping report assets")
    else:
        print("\n[7/7] Building report assets")
        from report_builder import build_report_assets

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
