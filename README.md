# Retina OCT Anomaly Detection Project

This repository implements the final-stage pipeline for the course project:

- train a convolutional autoencoder only on `NORMAL` retinal OCT scans,
- compare AE, VAE, L1, MSE+SSIM, latent-size, batch-size, crop/preprocessing, and extended-epoch ablation runs,
- estimate anomaly thresholds from validation reconstruction errors,
- detect pathological scans (`CNV`, `DME`, `DRUSEN`) through reconstruction-based anomaly scores,
- export figures, metrics, comparison tables, patient-level analysis, bootstrap confidence intervals, explainability grids, and report assets.

## Current final technical candidate

The strongest image-level setup is:

```text
ae_mse_l128_e60 + topk_mse_5
```

This uses a convolutional autoencoder trained only on normal OCT images for 60 epochs, then evaluates anomalies with the mean of the top 5% pixel-wise squared residuals.

| Evaluation level | Run / score | AUROC | F1 | Recall | Precision | FPR |
|---|---|---:|---:|---:|---:|---:|
| Image-level | `ae_mse_l128_e60 + topk_mse_5` | 0.9457 | 0.8464 | 0.7387 | 0.9911 | 0.0200 |
| Patient-level | `ae_mse_l128_e60 + mean(topk_mse_5)` | 0.9485 | 0.8975 | 0.8346 | 0.9706 | 0.0760 |

Important: the project intentionally keeps weaker trials too. VAE, L1 loss, MSE+SSIM loss, crop variants, latent-size ablation, batch-size ablation, and score ensembles are all preserved so the final report can discuss what was tried and what did not improve the baseline.

## Expected dataset layout

Place the Kermany OCT dataset under:

```text
data/
  oct2017/
    train/
      NORMAL/
      CNV/
      DME/
      DRUSEN/
    test/
      NORMAL/
      CNV/
      DME/
      DRUSEN/
```

The training pipeline uses only `train/NORMAL` for model fitting. Validation is created by patient-level splitting inside `train/NORMAL`. Testing uses every class under `test/`. Patient IDs preserve the class prefix (for example `NORMAL-0101` and `CNV-0101`) to avoid cross-class collisions during patient-level aggregation. The real dataset is intentionally ignored by Git.

## Environment

The code was prepared for the existing local Python environment at:

```text
..\odev2\.venv\Scripts\python.exe
```

Install dependencies if needed:

```powershell
..\odev2\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run a single experiment

```powershell
..\odev2\.venv\Scripts\python.exe main.py --run-id ae_mse_l128 --model-type ae --loss-type mse --latent-dim 128 --skip-report
```

Optional overrides:

```powershell
..\odev2\.venv\Scripts\python.exe main.py --data-root data/oct2017 --epochs 20 --batch-size 16 --num-workers 4
```

## Run the final experiment matrix

```powershell
..\odev2\.venv\Scripts\python.exe run_experiments.py --config configs/final_experiments.json --clean-outputs
```

This creates isolated run folders under:

```text
outputs/experiments/<run_id>/
```

Then it aggregates the final comparison outputs under:

```text
outputs/comparison/
```

Run the extended-epoch final candidate:

```powershell
..\odev2\.venv\Scripts\python.exe run_experiments.py --config configs/extended_epoch_ablation.json
```

Evaluate alternative anomaly scores for a completed checkpoint without retraining:

```powershell
..\odev2\.venv\Scripts\python.exe score_ablation.py --run-id ae_mse_l128_e60 --eval-batch-size 64 --num-workers 8
..\odev2\.venv\Scripts\python.exe compare_score_ablations.py
```

This creates `outputs/score_ablation/<run_id>/` with image-level metrics, patient-level metrics, bootstrap confidence intervals, threshold comparisons, class-wise summaries, ROC overlays, and top-k residual explainability grids.

The final candidate checkpoint is kept under `outputs/experiments/ae_mse_l128_e60/saved_models/best_autoencoder.pt` so this score ablation can be reproduced without retraining the best model. Other per-run checkpoints remain ignored by default to keep the repository smaller.

Build the technical experiment ledger used as a final-report checklist:

```powershell
..\odev2\.venv\Scripts\python.exe scripts/build_experiment_ledger.py
```

This writes `outputs/experiment_ledger.csv` and `report/experiment_ledger.md`. The ledger intentionally includes both successful and unsuccessful trials so the final report can mention every attempted direction: AE/VAE, L1, MSE+SSIM, latent-size ablation, batch-size ablation, crop/preprocessing trials, score ablation, patient-level evaluation, bootstrap confidence intervals, and explainability outputs.

## Smoke test without the real dataset

Generate a small synthetic dataset:

```powershell
..\odev2\.venv\Scripts\python.exe scripts/create_mock_oct_dataset.py
```

Then run:

```powershell
..\odev2\.venv\Scripts\python.exe main.py --data-root data/mock_oct2017 --run-id smoke_ae --model-type ae --loss-type mse --epochs 2 --batch-size 8 --num-workers 0 --output-root tmp/smoke_ae --skip-report --clean-outputs
..\odev2\.venv\Scripts\python.exe main.py --data-root data/mock_oct2017 --run-id smoke_vae --model-type vae --loss-type vae_mse_kl --epochs 2 --batch-size 8 --num-workers 0 --output-root tmp/smoke_vae --skip-report --clean-outputs
```

Run the lightweight unit checks:

```powershell
..\odev2\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Generated outputs

`outputs/` contains:

- `experiments/`: isolated experiment runs with metrics, figures, reconstruction examples, and run logs
- `comparison*/`: model, threshold, class-wise, batch-size, crop, and preprocessing comparison tables
- `score_ablation/`: image-level and patient-level score comparisons, bootstrap confidence intervals, ROC overlays, and top-k explainability grids
- `preprocessing/`: crop/preprocessing preview figures
- `experiment_ledger.csv`: compact record of all attempted experiments for final-report writing
- `summary.txt`, `metrics/`, `figures/`, `reconstructions/`, `saved_models/`: legacy baseline outputs kept for continuity

`report/` contains:

- `ara_rapor_draft.md`: markdown draft for the IEEE ara rapor
- `ara_rapor_draft.docx`: a DOCX draft built from the IEEE template when the template is available
- `literature_notes.md`: literature pool and positioning notes
- `report_context.json`: structured experiment data used to generate the report draft
- `experiment_ledger.md`: final technical experiment checklist, including both successful and unsuccessful trials
