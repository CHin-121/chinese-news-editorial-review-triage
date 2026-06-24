# Phase 1D: Calibrated Confidence Analysis

## Purpose

Phase 1D extends the existing editorial review triage analysis with probability calibration. Phase 1C ranked potentially risky predictions by the gap between the top two Linear SVM decision scores. That margin is useful as a ranking signal, but it is not directly interpretable as a probability.

This phase evaluates whether calibrated probabilities provide a more interpretable confidence signal for allocating limited human review resources. It is an offline diagnostic analysis, not evidence of real newsroom deployment.

## Models Compared

The analysis keeps the Phase 1B/1C TF-IDF workflow and compares:

- TF-IDF + Linear SVM raw decision margin;
- TF-IDF + Linear SVM with sigmoid calibration via `CalibratedClassifierCV`;
- TF-IDF + Logistic Regression as a probability baseline.

The calibrated Linear SVM uses Platt-style sigmoid calibration through scikit-learn's `CalibratedClassifierCV`. Logistic Regression is included because it provides native `predict_proba` outputs under the same TF-IDF representation.

## Metrics

Classification metrics are retained to check predictive performance:

- Accuracy;
- Macro-F1.

Calibration metrics are added to evaluate probabilistic confidence:

- Log Loss;
- multiclass Brier Score;
- top-label Expected Calibration Error (ECE);
- fixed-width confidence-bin summaries.

Top-label ECE is computed by grouping samples into confidence bins and comparing each bin's mean predicted confidence with empirical accuracy.

## Review Triage Comparison

The review triage curve compares:

- reviewing articles with the lowest raw Linear SVM margin;
- reviewing articles with the lowest calibrated top-label confidence;
- a random-review expected baseline.

The main question is not whether calibration increases classification accuracy. The question is whether calibrated confidence provides a more interpretable signal for review allocation.

If raw margin remains competitive for error capture, that is still a valid result: calibration improves probability interpretability, while margin may remain a strong ranking signal.

## Verified Phase 1D Results

Run ID: `20260623T215102Z`

The run used 40,000 training records and 10,000 held-out test records.

| Model | Accuracy | Macro-F1 | Log Loss | Brier Score | Top-label ECE |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Linear SVM + sigmoid calibration | 0.8261 | 0.8287 | 0.6004 | 0.2533 | 0.0529 |
| TF-IDF + Logistic Regression probability baseline | 0.8113 | 0.8154 | 0.8448 | 0.3542 | 0.2487 |

The calibrated Linear SVM produced substantially lower Log Loss, Brier Score, and ECE than the Logistic Regression probability baseline in this run.

Review triage results:

| Review budget | Raw SVM margin | Calibrated top-label confidence | Random expected |
|---:|---:|---:|---:|
| 5% | 18.57% | 20.39% | 5.00% |
| 10% | 35.72% | 36.97% | 10.00% |
| 20% | 60.08% | 61.50% | 20.00% |
| 30% | 77.91% | 78.65% | 30.00% |

In this offline run, calibrated confidence slightly outperformed the raw SVM margin in captured-error share across the tested review budgets. The practical interpretation should remain conservative: calibration makes confidence more interpretable and can support review prioritization, but it does not validate live deployment reliability.

## Outputs

Run outputs are written under `results/reproduced/<run_id>/`:

```text
calibration_metrics.csv
calibration_bins.csv
review_triage_calibrated_curve.csv
calibration_summary.json
```

Figures are written under `results/figures/`:

```text
calibration_reliability_diagram.png
confidence_histogram.png
review_triage_margin_vs_calibrated_confidence.png
```

The outputs do not include raw article text, per-article predictions, private local paths, or deployment data.

## Command

```bash
PYTHONPATH=src python -m repro_pipeline.cli \
  --data data/raw/cnews.train.txt \
  --historical-config configs/historical_compat.json \
  --clean-config configs/clean_baseline.json \
  --calibration-config configs/calibration.json \
  --calibration-only \
  --output-root results/reproduced \
  --figures-root results/figures
```

The raw dataset is not included in this repository. Anyone running the command must obtain the dataset separately and keep it outside Git.

## Limitations

- Calibrated probabilities are evaluated only on a held-out test split.
- The analysis is offline and does not validate live editorial deployment.
- The calibrated model is not claimed to improve Accuracy or Macro-F1 by default.
- Confidence estimates may shift under domain drift or a different news source.
- Raw decision margin and calibrated confidence answer related but distinct questions: ranking risk versus interpreting predicted probability.
