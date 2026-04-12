# Retina OCT Anomaly Detection Project

This repository implements the baseline pipeline for the course project:

- train a convolutional autoencoder only on `NORMAL` retinal OCT scans,
- estimate anomaly thresholds from validation reconstruction errors,
- detect pathological scans (`CNV`, `DME`, `DRUSEN`) through reconstruction error,
- export figures, metrics, and ara-rapor assets.

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

The training pipeline uses only `train/NORMAL` for model fitting. Validation is created by patient-level splitting inside `train/NORMAL`. Testing uses every class under `test/`.

## Environment

The code was prepared for the existing local Python environment at:

```text
..\odev2\.venv\Scripts\python.exe
```

Install dependencies if needed:

```powershell
..\odev2\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run with the real dataset

```powershell
..\odev2\.venv\Scripts\python.exe main.py --clean-outputs
```

Optional overrides:

```powershell
..\odev2\.venv\Scripts\python.exe main.py --data-root data/oct2017 --epochs 20 --batch-size 16
```

## Smoke test without the real dataset

Generate a small synthetic dataset:

```powershell
..\odev2\.venv\Scripts\python.exe scripts/create_mock_oct_dataset.py
```

Then run:

```powershell
..\odev2\.venv\Scripts\python.exe main.py --data-root data/mock_oct2017 --epochs 4 --batch-size 8 --clean-outputs
```

## Generated outputs

`outputs/` contains:

- `metrics/`: run config, CSV tables, threshold analysis, sample-level scores
- `figures/`: training loss, ROC, confusion matrix, error histograms
- `reconstructions/`: original vs reconstruction vs residual examples
- `saved_models/`: best autoencoder checkpoint
- `summary.txt`: one-page experiment summary

`report/` contains:

- `ara_rapor_draft.md`: markdown draft for the IEEE ara rapor
- `ara_rapor_draft.docx`: a DOCX draft built from the IEEE template when the template is available
- `literature_notes.md`: literature pool and positioning notes
- `report_context.json`: structured experiment data used to generate the report draft
