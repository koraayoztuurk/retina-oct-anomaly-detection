# Retina OCT Anomaly Detection Project

This repository implements the final-stage pipeline for the course project:

- train a convolutional autoencoder only on `NORMAL` retinal OCT scans,
- compare AE, VAE, L1, MSE+SSIM, latent-size, batch-size, crop/preprocessing, extended-epoch, learning-rate-scheduler, BatchNorm, and 256x256 high-resolution ablation runs,
- estimate anomaly thresholds from validation reconstruction errors,
- detect pathological scans (`CNV`, `DME`, `DRUSEN`) through reconstruction-based anomaly scores,
- export figures, metrics, comparison tables, patient-level analysis, bootstrap confidence intervals, and explainability grids.

## Current final technical candidate

The strongest image-level setup is:

```text
ae_mse_l128_e60_plateau_bn + topk_mse_5
```

This uses a convolutional autoencoder trained only on normal OCT images for 60 epochs with `BatchNorm2d` and `ReduceLROnPlateau`, then evaluates anomalies with the mean of the top 5% pixel-wise squared residuals.

| Evaluation level | Run / score | AUROC | F1 | Recall | Precision | FPR |
|---|---|---:|---:|---:|---:|---:|
| Image-level | `ae_mse_l128_e60_plateau_bn + topk_mse_5` | 0.9487 | 0.8593 | 0.7613 | 0.9862 | 0.0320 |
| Patient-level | `ae_mse_l128_e60_plateau_bn + mean(topk_mse_5)` | 0.9513 | 0.9089 | 0.8541 | 0.9712 | 0.0760 |

Important: the project intentionally keeps weaker trials too. VAE, L1 loss, MSE+SSIM loss, crop variants, latent-size ablation, batch-size ablation, fixed-LR extended training, learning-rate scheduling, BatchNorm, 256x256 high-resolution training, and score ensembles are all preserved so the final report can discuss what was tried and what did not improve the baseline.

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
..\odev2\.venv\Scripts\python.exe main.py --run-id ae_mse_l128 --model-type ae --loss-type mse --latent-dim 128
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

Run the learning-rate-scheduler candidate:

```powershell
..\odev2\.venv\Scripts\python.exe run_experiments.py --config configs/scheduler_ablation.json --clean-outputs --write-logs --aggregate-config-only --comparison-root outputs/comparison_scheduler
```

Run the BatchNorm candidate:

```powershell
..\odev2\.venv\Scripts\python.exe run_experiments.py --config configs/batchnorm_ablation.json --clean-outputs --write-logs --aggregate-config-only --comparison-root outputs/comparison_batchnorm
```

Run the 256x256 high-resolution ablation:

```powershell
..\odev2\.venv\Scripts\python.exe run_experiments.py --config configs/highres_ablation.json --clean-outputs --write-logs --aggregate-config-only --comparison-root outputs/comparison_highres
```

Evaluate alternative anomaly scores for a completed checkpoint without retraining:

```powershell
..\odev2\.venv\Scripts\python.exe score_ablation.py --run-id ae_mse_l128_e60_plateau_bn --eval-batch-size 128 --num-workers 0
..\odev2\.venv\Scripts\python.exe compare_score_ablations.py
```

This creates `outputs/score_ablation/<run_id>/` with image-level metrics, patient-level metrics, bootstrap confidence intervals, threshold comparisons, class-wise summaries, ROC overlays, and top-k residual explainability grids.

The final candidate checkpoint is kept under `outputs/experiments/ae_mse_l128_e60_plateau_bn/saved_models/best_autoencoder.pt` so this score ablation can be reproduced without retraining the best model. Other per-run checkpoints remain ignored by default to keep the repository smaller.

Build the technical experiment ledger used as a final-report checklist:

```powershell
..\odev2\.venv\Scripts\python.exe scripts/build_experiment_ledger.py
```

This writes `outputs/experiment_ledger.csv`. The ledger intentionally includes both successful and unsuccessful trials so the final report can mention every attempted direction: AE/VAE, L1, MSE+SSIM, latent-size ablation, batch-size ablation, crop/preprocessing trials, learning-rate scheduling, BatchNorm, 256x256 high-resolution training, score ablation, patient-level evaluation, bootstrap confidence intervals, and explainability outputs.

## Smoke test without the real dataset

Generate a small synthetic dataset:

```powershell
..\odev2\.venv\Scripts\python.exe scripts/create_mock_oct_dataset.py
```

Then run:

```powershell
..\odev2\.venv\Scripts\python.exe main.py --data-root data/mock_oct2017 --run-id smoke_ae --model-type ae --loss-type mse --epochs 2 --batch-size 8 --num-workers 0 --output-root tmp/smoke_ae --clean-outputs
..\odev2\.venv\Scripts\python.exe main.py --data-root data/mock_oct2017 --run-id smoke_vae --model-type vae --loss-type vae_mse_kl --epochs 2 --batch-size 8 --num-workers 0 --output-root tmp/smoke_vae --clean-outputs
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

- `Grup12_KorayÖztürk_EmirAlpİlhan.pdf`: submitted ara rapor
- `Grup12_KorayÖztürk_EmirAlpİlhan_Final_Rapor.pdf`: submitted final report
- `Grup12_KorayÖztürk_EmirAlpİlhan_Final_Rapor.docx`: editable final report
