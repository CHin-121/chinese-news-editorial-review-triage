"""Phase 1D calibrated confidence analysis for review triage.

The functions in this module keep text and per-article outputs out of saved
artifacts. Public outputs are aggregate metrics, confidence bins, triage curves,
figures, and summary metadata only.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .classification import _split_and_encode, _vectorizer
from .reporting import assert_metadata_private


def _validate_probability_inputs(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a 2D array.")
    if len(y_true) != probabilities.shape[0]:
        raise ValueError("Labels and probabilities must have the same row count.")
    if probabilities.shape[1] < 2:
        raise ValueError("At least two probability columns are required.")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be in the interval [0, 1].")
    return y_true, probabilities


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the multiclass Brier score against one-hot encoded targets."""
    y_true, probabilities = _validate_probability_inputs(y_true, probabilities)
    class_count = probabilities.shape[1]
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    squared_error = np.square(probabilities - one_hot).sum(axis=1)
    return float(np.mean(squared_error))


def build_confidence_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Aggregate top-label confidence calibration by fixed-width bins."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    y_true, probabilities = _validate_probability_inputs(y_true, probabilities)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == y_true
    rows: list[dict[str, float | int]] = []

    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        sample_count = int(mask.sum())
        if sample_count:
            mean_confidence = float(confidence[mask].mean())
            empirical_accuracy = float(correct[mask].mean())
            calibration_gap = abs(empirical_accuracy - mean_confidence)
        else:
            mean_confidence = float("nan")
            empirical_accuracy = float("nan")
            calibration_gap = float("nan")
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "sample_count": sample_count,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "calibration_gap": calibration_gap,
            }
        )
    return rows


def top_label_ece(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Return top-label Expected Calibration Error using fixed-width bins."""
    y_true, probabilities = _validate_probability_inputs(y_true, probabilities)
    bins = build_confidence_bins(y_true, probabilities, n_bins=n_bins)
    total = len(y_true)
    ece = 0.0
    for row in bins:
        sample_count = int(row["sample_count"])
        if not sample_count:
            continue
        ece += sample_count / total * float(row["calibration_gap"])
    return float(ece)


def _review_rows_for_order(
    *,
    strategy: str,
    order: np.ndarray,
    error_mask: np.ndarray,
    budgets: tuple[float, ...],
) -> list[dict[str, float | int | str]]:
    total_samples = len(error_mask)
    total_errors = int(error_mask.sum())
    rows: list[dict[str, float | int | str]] = []
    for budget in budgets:
        if not 0 < budget <= 1:
            raise ValueError("Review budgets must be in the interval (0, 1].")
        reviewed_count = min(total_samples, math.ceil(total_samples * budget))
        reviewed_indices = order[:reviewed_count]
        captured_error_count = int(error_mask[reviewed_indices].sum())
        rows.append(
            {
                "strategy": strategy,
                "budget_percent": budget * 100.0,
                "reviewed_count": reviewed_count,
                "captured_error_count": captured_error_count,
                "share_of_all_errors_captured": (
                    captured_error_count / total_errors if total_errors else 0.0
                ),
                "errors_found_per_100_reviewed": (
                    captured_error_count / reviewed_count * 100.0
                    if reviewed_count
                    else 0.0
                ),
                "total_errors": total_errors,
            }
        )
    return rows


def build_review_triage_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_margins: np.ndarray,
    calibrated_probabilities: np.ndarray,
    *,
    budgets: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30),
) -> list[dict[str, float | int | str]]:
    """Compare error capture under margin, calibrated confidence, and random."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    raw_margins = np.asarray(raw_margins, dtype=float)
    _, calibrated_probabilities = _validate_probability_inputs(
        y_true, calibrated_probabilities
    )
    if len(y_true) != len(y_pred) or len(y_true) != len(raw_margins):
        raise ValueError("Labels, predictions, and raw margins must align.")

    error_mask = y_true != y_pred
    margin_order = np.argsort(raw_margins, kind="stable")
    calibrated_confidence = calibrated_probabilities.max(axis=1)
    confidence_order = np.argsort(calibrated_confidence, kind="stable")
    rows = [
        *_review_rows_for_order(
            strategy="raw_svm_margin",
            order=margin_order,
            error_mask=error_mask,
            budgets=budgets,
        ),
        *_review_rows_for_order(
            strategy="calibrated_top_label_confidence",
            order=confidence_order,
            error_mask=error_mask,
            budgets=budgets,
        ),
    ]

    total_samples = len(y_true)
    total_errors = int(error_mask.sum())
    for budget in budgets:
        reviewed_count = min(total_samples, math.ceil(total_samples * budget))
        expected_errors = reviewed_count * total_errors / total_samples
        rows.append(
            {
                "strategy": "random_expected",
                "budget_percent": budget * 100.0,
                "reviewed_count": reviewed_count,
                "captured_error_count": float(expected_errors),
                "share_of_all_errors_captured": (
                    expected_errors / total_errors if total_errors else 0.0
                ),
                "errors_found_per_100_reviewed": (
                    expected_errors / reviewed_count * 100.0
                    if reviewed_count
                    else 0.0
                ),
                "total_errors": total_errors,
            }
        )
    return rows


def _top_two_margin(decision_scores: np.ndarray) -> np.ndarray:
    top_two = np.partition(decision_scores, -2, axis=1)[:, -2:]
    return top_two.max(axis=1) - top_two.min(axis=1)


def _probability_metric_rows(
    *,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int,
) -> list[dict[str, Any]]:
    class_count = probabilities.shape[1]
    labels = np.arange(class_count)
    return [
        {
            "model": model_name,
            "metric": "Accuracy",
            "value": float(accuracy_score(y_true, y_pred)),
            "unit": "ratio",
        },
        {
            "model": model_name,
            "metric": "Macro-F1",
            "value": float(f1_score(y_true, y_pred, average="macro")),
            "unit": "ratio",
        },
        {
            "model": model_name,
            "metric": "Log Loss",
            "value": float(log_loss(y_true, probabilities, labels=labels)),
            "unit": "loss",
        },
        {
            "model": model_name,
            "metric": "Multiclass Brier Score",
            "value": multiclass_brier_score(y_true, probabilities),
            "unit": "score",
        },
        {
            "model": model_name,
            "metric": "Top-label ECE",
            "value": top_label_ece(y_true, probabilities, n_bins=n_bins),
            "unit": "score",
        },
    ]


def run_calibrated_confidence_analysis(
    texts: list[str],
    labels: list[str],
    clean_config: dict[str, Any],
    calibration_config: dict[str, Any],
    *,
    selected_c: float,
) -> dict[str, Any]:
    """Run Phase 1D calibration analysis using aggregate held-out outputs."""
    X_train, X_test, y_train, y_test, classes = _split_and_encode(
        texts, labels, clean_config
    )
    linear_svm_config = clean_config["models"]["Linear SVM"]
    logistic_config = clean_config["models"]["Logistic Regression"]
    evaluation = calibration_config["evaluation"]
    n_bins = int(evaluation["confidence_bins"])
    budgets = tuple(float(value) for value in evaluation["review_budgets"])

    raw_svm = Pipeline(
        [
            ("tfidf", _vectorizer(clean_config)),
            (
                "classifier",
                LinearSVC(
                    C=selected_c,
                    dual=linear_svm_config["dual"],
                    random_state=linear_svm_config["random_state"],
                ),
            ),
        ]
    )
    raw_svm.fit(X_train, y_train)
    raw_prediction = raw_svm.predict(X_test)
    raw_margins = _top_two_margin(raw_svm.decision_function(X_test))

    calibrated_svm = CalibratedClassifierCV(
        estimator=Pipeline(
            [
                ("tfidf", _vectorizer(clean_config)),
                (
                    "classifier",
                    LinearSVC(
                        C=selected_c,
                        dual=linear_svm_config["dual"],
                        random_state=linear_svm_config["random_state"],
                    ),
                ),
            ]
        ),
        method=calibration_config["model"]["calibration_method"],
        cv=calibration_config["model"]["cv"],
        ensemble=calibration_config["model"]["ensemble"],
    )
    calibrated_svm.fit(X_train, y_train)
    calibrated_probabilities = calibrated_svm.predict_proba(X_test)
    calibrated_prediction = calibrated_probabilities.argmax(axis=1)

    logistic = Pipeline(
        [
            ("tfidf", _vectorizer(clean_config)),
            (
                "classifier",
                LogisticRegression(
                    C=selected_c,
                    solver=logistic_config["solver"],
                    max_iter=logistic_config["max_iter"],
                    random_state=logistic_config["random_state"],
                ),
            ),
        ]
    )
    logistic.fit(X_train, y_train)
    logistic_probabilities = logistic.predict_proba(X_test)
    logistic_prediction = logistic_probabilities.argmax(axis=1)

    metrics = [
        *_probability_metric_rows(
            model_name="TF-IDF + Linear SVM + sigmoid calibration",
            y_true=y_test,
            y_pred=calibrated_prediction,
            probabilities=calibrated_probabilities,
            n_bins=n_bins,
        ),
        *_probability_metric_rows(
            model_name="TF-IDF + Logistic Regression probability baseline",
            y_true=y_test,
            y_pred=logistic_prediction,
            probabilities=logistic_probabilities,
            n_bins=n_bins,
        ),
    ]
    bins = [
        {
            "model": "TF-IDF + Linear SVM + sigmoid calibration",
            **row,
        }
        for row in build_confidence_bins(y_test, calibrated_probabilities, n_bins=n_bins)
    ]
    bins.extend(
        {
            "model": "TF-IDF + Logistic Regression probability baseline",
            **row,
        }
        for row in build_confidence_bins(y_test, logistic_probabilities, n_bins=n_bins)
    )
    triage_curve = build_review_triage_comparison(
        y_test,
        raw_prediction,
        raw_margins,
        calibrated_probabilities,
        budgets=budgets,
    )
    summary = {
        "phase": "Phase 1D",
        "analysis": "Calibrated Confidence Analysis",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "class_labels": classes,
        "selected_c": selected_c,
        "calibration_method": calibration_config["model"]["calibration_method"],
        "calibration_cv": calibration_config["model"]["cv"],
        "confidence_bins": n_bins,
        "review_budgets": list(budgets),
        "raw_margin_definition": "top1_decision_score - top2_decision_score",
        "calibrated_confidence_definition": "max predicted class probability",
        "scope_note": (
            "Offline held-out calibration diagnostic; no raw article text, "
            "per-article predictions, or deployment claim is saved."
        ),
        "raw_svm_accuracy": float(accuracy_score(y_test, raw_prediction)),
        "raw_svm_macro_f1": float(f1_score(y_test, raw_prediction, average="macro")),
    }
    return {
        "metrics": metrics,
        "confidence_bins": bins,
        "review_triage_curve": triage_curve,
        "summary": summary,
        "figures": {
            "calibrated_probabilities": calibrated_probabilities,
            "logistic_probabilities": logistic_probabilities,
            "y_true": y_test,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path.name}.")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_reliability_diagram(
    path: Path,
    bins: list[dict[str, Any]],
) -> None:
    plt.figure(figsize=(7, 5))
    for model_name in sorted({row["model"] for row in bins}):
        model_rows = [
            row
            for row in bins
            if row["model"] == model_name and not math.isnan(float(row["mean_confidence"]))
        ]
        x_values = [float(row["mean_confidence"]) for row in model_rows]
        y_values = [float(row["empirical_accuracy"]) for row in model_rows]
        plt.plot(x_values, y_values, marker="o", label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability Diagram")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_confidence_histogram(
    path: Path,
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    plt.figure(figsize=(7, 5))
    plt.hist(confidence[correct], bins=10, alpha=0.65, label="Correct")
    plt.hist(confidence[~correct], bins=10, alpha=0.65, label="Incorrect")
    plt.xlabel("Top-label confidence")
    plt.ylabel("Sample count")
    plt.title("Confidence Distribution: Calibrated Linear SVM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_triage_curve(
    path: Path,
    triage_rows: list[dict[str, Any]],
) -> None:
    plt.figure(figsize=(7, 5))
    for strategy in ["raw_svm_margin", "calibrated_top_label_confidence", "random_expected"]:
        rows = [row for row in triage_rows if row["strategy"] == strategy]
        rows.sort(key=lambda row: float(row["budget_percent"]))
        plt.plot(
            [float(row["budget_percent"]) for row in rows],
            [float(row["share_of_all_errors_captured"]) * 100.0 for row in rows],
            marker="o",
            label=strategy,
        )
    plt.xlabel("Review budget (%)")
    plt.ylabel("Share of all errors captured (%)")
    plt.title("Review Triage: Margin vs Calibrated Confidence")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_calibration_outputs(
    output_root: str | Path,
    figures_root: str | Path,
    run_id: str,
    analysis: dict[str, Any],
) -> Path:
    """Write aggregate Phase 1D outputs and figures."""
    summary = analysis["summary"]
    assert_metadata_private(summary)
    run_directory = Path(output_root) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    figures_directory = Path(figures_root)
    figures_directory.mkdir(parents=True, exist_ok=True)

    _write_csv(run_directory / "calibration_metrics.csv", analysis["metrics"])
    _write_csv(run_directory / "calibration_bins.csv", analysis["confidence_bins"])
    _write_csv(run_directory / "review_triage_calibrated_curve.csv", analysis["review_triage_curve"])
    with (run_directory / "calibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    figure_payload = analysis["figures"]
    _plot_reliability_diagram(
        figures_directory / "calibration_reliability_diagram.png",
        analysis["confidence_bins"],
    )
    _plot_confidence_histogram(
        figures_directory / "confidence_histogram.png",
        y_true=figure_payload["y_true"],
        probabilities=figure_payload["calibrated_probabilities"],
    )
    _plot_triage_curve(
        figures_directory / "review_triage_margin_vs_calibrated_confidence.png",
        analysis["review_triage_curve"],
    )
    return run_directory
