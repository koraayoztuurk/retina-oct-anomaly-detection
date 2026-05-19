from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "outputs" / "experiments"
SCORE_ROOT = PROJECT_ROOT / "outputs" / "score_ablation"
PREPROCESSING_ROOT = PROJECT_ROOT / "outputs" / "preprocessing"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
REPORT_ROOT = PROJECT_ROOT / "report"


RUN_NOTES = {
    "ae_mse_l128": "Ana 40 epoch AE-MSE baseline; score ablation ile güçlü sonuç verdi ancak 60 epoch denemesi tarafından geçildi.",
    "ae_mse_l128_e60": "AE-MSE latent 128 için 60 epoch denemesi; validation loss ve top-k score metriklerinde 40 epoch koşusunu iyileştirdi.",
    "ae_mse_l128_e60_plateau": "AE-MSE 60 epoch + ReduceLROnPlateau denemesi; validation loss'u düşürdü ve top-k score ile yeni en iyi final adayını verdi.",
    "vae_msekl_l128": "VAE + KL denemesi; generative latent model beklenen iyileştirmeyi sağlamadı.",
    "ae_l1_l128": "L1 loss denemesi; MSE baseline'a göre belirgin zayıf kaldı.",
    "ae_mse_ssim_l128": "MSE+SSIM training loss denemesi; bu veri ve ayarda zayıf kaldı.",
    "ae_mse_l64": "Latent 64 ablasyonu; MSE skorunda güçlü ama top-k skorla l128/l256 gerisinde.",
    "ae_mse_l256": "Latent 256 ablasyonu; MSE skorunda güçlü, top-k skorla da stabil.",
    "ae_mse_l256_bs16": "Batch size 16 denemesi; MSE'de güçlü, top-k AUROC en yükseklerden biri.",
    "ae_mse_l256_bs64": "Batch size 64 denemesi; batch size 16/32'ye göre daha zayıf.",
    "ae_mse_l256_bs16_crop": "Content crop denemesi; DRUSEN biraz artsa da genel performans düşük.",
    "ae_mse_l256_bs32_retina_margin": "Retina margin crop denemesi; DRUSEN yakalama arttı ancak FPR ve genel metrikler zayıfladı.",
}


PREPROCESSING_NOTES = [
    {
        "category": "preprocessing_preview",
        "run_id": "content_crop_preview",
        "setting": "SafeContentCrop",
        "status": "preview + full training",
        "artifact": "outputs/preprocessing/content_crop_preview.png",
        "note": "İlk içerik tabanlı kırpma bazı görüntülerde fazla agresif bulundu; yine de full training denendi ve genel performansı düşürdü.",
    },
    {
        "category": "preprocessing_preview",
        "run_id": "border_crop_preview",
        "setting": "ConservativeBorderCrop",
        "status": "preview only",
        "artifact": "outputs/preprocessing/border_crop_preview.png",
        "note": "Sadece kenar boşluklarını azaltan daha konservatif kırpma önizlemesi üretildi; full training'e alınmadı.",
    },
    {
        "category": "preprocessing_preview",
        "run_id": "retina_margin_crop_preview",
        "setting": "RetinaMarginCrop",
        "status": "preview + full training",
        "artifact": "outputs/preprocessing/retina_margin_crop_preview.png",
        "note": "Retina sinyaline göre üst/alt boşluk bırakan kırpma denendi; DRUSEN yakalama arttı fakat genel metrikler baseline'ın gerisinde kaldı.",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def format_float(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def drusen_detection(run_dir: Path) -> tuple[Any, Any]:
    classwise_df = read_csv(run_dir / "metrics" / "classwise_reconstruction_summary.csv")
    if classwise_df.empty or "class_name" not in classwise_df.columns:
        return "", ""
    drusen = classwise_df[classwise_df["class_name"] == "DRUSEN"]
    if drusen.empty:
        return "", ""
    return drusen.iloc[0].get("detected_count", ""), drusen.iloc[0].get("detection_rate", "")


def collect_training_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not EXPERIMENTS_ROOT.exists():
        return rows

    for run_dir in sorted(path for path in EXPERIMENTS_ROOT.iterdir() if path.is_dir()):
        config = read_json(run_dir / "metrics" / "run_config.json")
        metrics = read_json(run_dir / "metrics" / "selected_threshold_metrics.json")
        dataset_checks = read_json(run_dir / "metrics" / "dataset_checks.json")
        drusen_count, drusen_rate = drusen_detection(run_dir)
        rows.append(
            {
                "category": "training_run",
                "run_id": run_dir.name,
                "model_type": config.get("model_type", metrics.get("model_type", "")),
                "loss_type": config.get("loss_type", metrics.get("loss_type", "")),
                "score_mode": metrics.get("score_type", "mse"),
                "latent_dim": config.get("latent_dim", ""),
                "batch_size": config.get("batch_size", ""),
                "crop_mode": config.get("crop_mode", "none"),
                "image_size": config.get("image_size", ""),
                "threshold_percentile": metrics.get("default_percentile", config.get("default_percentile", "")),
                "auroc": metrics.get("auroc", ""),
                "accuracy": metrics.get("accuracy", ""),
                "precision": metrics.get("precision", ""),
                "recall": metrics.get("recall", ""),
                "f1": metrics.get("f1", ""),
                "fpr": metrics.get("fpr", ""),
                "tn": metrics.get("tn", ""),
                "fp": metrics.get("fp", ""),
                "fn": metrics.get("fn", ""),
                "tp": metrics.get("tp", ""),
                "drusen_detected": drusen_count,
                "drusen_detection_rate": drusen_rate,
                "train_pathology_count": dataset_checks.get("train_pathology_count", ""),
                "train_val_patient_overlap": dataset_checks.get("train_val_patient_overlap_count", ""),
                "artifact": f"outputs/experiments/{run_dir.name}",
                "note": RUN_NOTES.get(run_dir.name, ""),
            }
        )
    return rows


def collect_score_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_by_run = read_csv(SCORE_ROOT / "best_score_by_run.csv")
    if not best_by_run.empty:
        for _, item in best_by_run.iterrows():
            rows.append(
                {
                    "category": "score_ablation_image_level",
                    "run_id": item.get("run_id", ""),
                    "model_type": "",
                    "loss_type": "",
                    "score_mode": item.get("score_mode", ""),
                    "latent_dim": "",
                    "batch_size": "",
                    "crop_mode": "",
                    "image_size": "",
                    "threshold_percentile": 95,
                    "auroc": item.get("auroc", ""),
                    "accuracy": item.get("accuracy", ""),
                    "precision": item.get("precision", ""),
                    "recall": item.get("recall", ""),
                    "f1": item.get("f1", ""),
                    "fpr": item.get("fpr", ""),
                    "tn": item.get("tn", ""),
                    "fp": item.get("fp", ""),
                    "fn": item.get("fn", ""),
                    "tp": item.get("tp", ""),
                    "drusen_detected": "",
                    "drusen_detection_rate": "",
                    "train_pathology_count": "",
                    "train_val_patient_overlap": "",
                    "artifact": f"outputs/score_ablation/{item.get('run_id', '')}",
                    "note": "Mevcut checkpoint yeniden eğitilmeden farklı anomaly score'lar ile değerlendirildi; top-k residual skorlar en güçlü çıktı.",
                }
            )

    best_patient = read_csv(SCORE_ROOT / "best_patient_score_by_run.csv")
    if not best_patient.empty:
        for _, item in best_patient.iterrows():
            rows.append(
                {
                    "category": "score_ablation_patient_level",
                    "run_id": item.get("run_id", ""),
                    "model_type": "",
                    "loss_type": "",
                    "score_mode": item.get("score_mode", ""),
                    "latent_dim": "",
                    "batch_size": "",
                    "crop_mode": "",
                    "image_size": "",
                    "threshold_percentile": 95,
                    "auroc": item.get("auroc", ""),
                    "accuracy": item.get("accuracy", ""),
                    "precision": item.get("precision", ""),
                    "recall": item.get("recall", ""),
                    "f1": item.get("f1", ""),
                    "fpr": item.get("fpr", ""),
                    "tn": item.get("tn", ""),
                    "fp": item.get("fp", ""),
                    "fn": item.get("fn", ""),
                    "tp": item.get("tp", ""),
                    "drusen_detected": "",
                    "drusen_detection_rate": "",
                    "train_pathology_count": "",
                    "train_val_patient_overlap": "",
                    "artifact": f"outputs/score_ablation/{item.get('run_id', '')}/patient_level_comparison.csv",
                    "note": f"Hasta seviyesinde {item.get('patient_aggregation', '')} aggregation ile hesaplandı.",
                }
            )
    return rows


def collect_preprocessing_notes() -> list[dict[str, Any]]:
    rows = []
    for item in PREPROCESSING_NOTES:
        artifact_path = PROJECT_ROOT / item["artifact"]
        rows.append(
            {
                "category": item["category"],
                "run_id": item["run_id"],
                "model_type": "",
                "loss_type": "",
                "score_mode": "",
                "latent_dim": "",
                "batch_size": "",
                "crop_mode": item["setting"],
                "image_size": "",
                "threshold_percentile": "",
                "auroc": "",
                "accuracy": "",
                "precision": "",
                "recall": "",
                "f1": "",
                "fpr": "",
                "tn": "",
                "fp": "",
                "fn": "",
                "tp": "",
                "drusen_detected": "",
                "drusen_detection_rate": "",
                "train_pathology_count": "",
                "train_val_patient_overlap": "",
                "artifact": item["artifact"] if artifact_path.exists() else "",
                "note": item["note"],
            }
        )
    return rows


def build_markdown(ledger_df: pd.DataFrame) -> str:
    best_image = read_json(SCORE_ROOT / "best_score_overall.json")
    best_patient = read_json(SCORE_ROOT / "best_patient_score_overall.json")
    lines = [
        "# Final Teknik Deney Envanteri",
        "",
        "Bu dosya final raporu yazılırken hiçbir denemenin unutulmaması için tutulmuştur. Başarısız veya zayıf kalan denemeler de raporda kısaca belirtilmelidir.",
        "",
        "## Sabit Deney Kuralları",
        "",
        "- Eğitimde yalnızca `train/NORMAL` kullanıldı; patolojik sınıflar train'e alınmadı.",
        "- Validation ayrımı hasta ID tabanlı yapıldı; train/validation hasta kesişmesi kontrol edildi.",
        "- Eşik seçimi test setinden değil, validation normal skor dağılımından yapıldı.",
        "- Ana problem binary anomaly detection olarak tutuldu: `NORMAL` vs `CNV/DME/DRUSEN`.",
        "- Ana operasyon eşiği p95 olarak kullanıldı; p97/p99 karşılaştırma tabloları da saklandı.",
        "",
        "## En İyi Adaylar",
        "",
    ]
    if best_image:
        lines.append(
            f"- Image-level final adayı: `{best_image.get('run_id')}` + `{best_image.get('score_mode')}`; "
            f"AUROC={format_float(best_image.get('auroc'))}, F1={format_float(best_image.get('f1'))}, "
            f"Recall={format_float(best_image.get('recall'))}, Precision={format_float(best_image.get('precision'))}, FPR={format_float(best_image.get('fpr'))}."
        )
    if best_patient:
        lines.append(
            f"- Patient-level final adayı: `{best_patient.get('run_id')}` + `{best_patient.get('patient_aggregation')}` aggregation + `{best_patient.get('score_mode')}`; "
            f"AUROC={format_float(best_patient.get('auroc'))}, F1={format_float(best_patient.get('f1'))}, "
            f"Recall={format_float(best_patient.get('recall'))}, Precision={format_float(best_patient.get('precision'))}, FPR={format_float(best_patient.get('fpr'))}."
        )

    lines.extend(
        [
            "",
            "## Rapor Checklist",
            "",
            "- AE-MSE baseline anlatılacak.",
            "- VAE + KL denemesi anlatılacak; beklenen iyileştirmeyi sağlamadığı saklanmayacak.",
            "- L1 loss ve MSE+SSIM loss denemeleri anlatılacak; SSIM training loss'un zayıf kaldığı belirtilecek.",
            "- Latent size ablasyonu: 64/128/256 karşılaştırılacak.",
            "- Batch size ablasyonu: 16/32/64 karşılaştırılacak.",
            "- Learning rate scheduler denemesi anlatılacak: ReduceLROnPlateau sabit LR 60 epoch koşusunu küçük ama tutarlı biçimde iyileştirdi.",
            "- Crop/preprocessing denemeleri anlatılacak: content crop ve retina margin crop full run; border crop preview.",
            "- Score ablation anlatılacak: MSE, L1, SSIM score, retina-band, weighted retina, top-k residual ve ensemble skorlar.",
            "- Patient-level evaluation anlatılacak.",
            "- Bootstrap confidence interval anlatılacak.",
            "- Top-k residual explainability ve DRUSEN false negative örnekleri anlatılacak.",
            "",
            "## Deney Özeti",
            "",
            "| Kategori | Run | Model | Loss | Score | Latent | Batch | Crop | AUROC | F1 | Recall | Precision | FPR | DRUSEN | Not |",
            "|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for _, row in ledger_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("category", "")),
                    str(row.get("run_id", "")),
                    str(row.get("model_type", "")),
                    str(row.get("loss_type", "")),
                    str(row.get("score_mode", "")),
                    str(row.get("latent_dim", "")),
                    str(row.get("batch_size", "")),
                    str(row.get("crop_mode", "")),
                    format_float(row.get("auroc")),
                    format_float(row.get("f1")),
                    format_float(row.get("recall")),
                    format_float(row.get("precision")),
                    format_float(row.get("fpr")),
                    str(row.get("drusen_detected", "")),
                    str(row.get("note", "")).replace("|", "/"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Saklanan Ana Artefaktlar",
            "",
            "- `outputs/experiments/<run_id>/`: her training run için metrikler, figürler, reconstruction örnekleri ve config.",
            "- `outputs/comparison*/`: model, threshold ve class-wise karşılaştırma tabloları.",
            "- `outputs/score_ablation/`: score ablation, patient-level metrikler, bootstrap CI ve top-k explainability çıktısı.",
            "- `outputs/preprocessing/`: crop önizleme görselleri.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = collect_training_runs()
    rows.extend(collect_score_runs())
    rows.extend(collect_preprocessing_notes())

    ledger_df = pd.DataFrame(rows)
    if not ledger_df.empty:
        category_order = {
            "training_run": 0,
            "score_ablation_image_level": 1,
            "score_ablation_patient_level": 2,
            "preprocessing_preview": 3,
        }
        ledger_df["_category_order"] = ledger_df["category"].map(category_order).fillna(99)
        ledger_df = ledger_df.sort_values(["_category_order", "run_id"]).drop(columns=["_category_order"])

    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUTS_ROOT / "experiment_ledger.csv"
    md_path = REPORT_ROOT / "experiment_ledger.md"
    ledger_df.to_csv(csv_path, index=False)
    md_path.write_text(build_markdown(ledger_df), encoding="utf-8")

    print(f"[INFO] Saved CSV ledger : {csv_path}")
    print(f"[INFO] Saved MD ledger  : {md_path}")
    print(f"[INFO] Ledger rows      : {len(ledger_df)}")


if __name__ == "__main__":
    main()
