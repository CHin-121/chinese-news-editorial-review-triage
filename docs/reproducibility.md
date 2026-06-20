# Phase 1B-minimal Reproducibility Guide

## Verified Local Dataset

The audit used a local `cnews.train.txt` file that is not stored in this repository.

| Property | Verified value |
|---|---:|
| Size | 130,089,129 bytes |
| SHA-256 | `7e8710f786b03f8e7cad7230141446a405757f335d91c2ed64187cfaf30c6330` |
| Encoding | UTF-8 |
| Record format | `label<TAB>text` |
| Valid records | 50,000 |
| Classes | 10 |
| Records per class | 5,000 |
| Invalid records | 0 |

No article text, sample text, or private dataset path is written to the generated results.

## Reproduced Run

Run ID: `20260619T193149Z`

Outputs:

```text
results/reproduced/20260619T193149Z/
├── metrics.csv
├── comparison.csv
└── run_metadata.json
```

The result directory contains regenerated aggregate metrics, comparisons with the five approved historical values, and environment/configuration metadata. It contains no predictions or news text.

## Environment Used

```text
Python 3.12.13
numpy 2.2.6
pandas 2.3.3
scikit-learn 1.7.2
scipy 1.15.3
joblib 1.5.3
```

The historical project environment was no longer runnable, so the audit used an isolated temporary environment with the dependency versions recorded in the course project. The temporary environment is not part of the repository.

## Running the Minimal Audit

Obtain the dataset legally and keep it outside Git. From the repository root, run:

```bash
PYTHONPATH=src python -m repro_pipeline.cli \
  --data data/raw/cnews.train.txt \
  --historical-config configs/historical_compat.json \
  --clean-config configs/clean_baseline.json \
  --output-root results/reproduced
```

Before training, the command verifies the exact file size, SHA-256, UTF-8 decoding, and `label<TAB>text` structure. A mismatch stops execution. Results are written only after data validation and all three experiment components finish successfully.

## Result Separation

- `results/metrics.csv` remains the unchanged Phase 1A transcription of historical report values.
- `results/figures/` remains the unchanged Phase 1A archive of historical figures.
- Every new audit run is written to `results/reproduced/<run_id>/`.
- Regenerated rows are labelled by `result_type`; they must not replace or be presented as the historical report rows.

## Verification Status Rules

- `exact_at_reported_precision`: regenerated value equals the historical value after rounding to two decimal places for percentages or four decimal places for NMI.
- `near_match`: the classification difference is no more than 0.10 percentage points, or the NMI difference is no more than 0.0020.
- `not_reproduced`: the difference exceeds the corresponding tolerance.

Under run `20260619T193149Z`, all four classification metrics are exact at reported precision. Direct TF-IDF + KMeans is not reproduced.

## Explicit Non-Scope

This minimal release does not run or correct Word2Vec, Doc2Vec, SVD + KMeans, DBSCAN, or BERT. It also does not regenerate the archived figures or claim full end-to-end reproducibility.
