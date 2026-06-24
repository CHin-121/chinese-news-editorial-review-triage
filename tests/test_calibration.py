import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from repro_pipeline.calibration import (
    build_confidence_bins,
    build_review_triage_comparison,
    multiclass_brier_score,
    run_calibrated_confidence_analysis,
    top_label_ece,
    write_calibration_outputs,
)


class CalibrationMetricTests(unittest.TestCase):
    def test_multiclass_brier_score_uses_one_hot_targets(self):
        y_true = np.array([0, 1])
        probabilities = np.array(
            [
                [0.80, 0.20],
                [0.30, 0.70],
            ]
        )

        score = multiclass_brier_score(y_true, probabilities)

        self.assertAlmostEqual(score, 0.13)

    def test_top_label_ece_weights_bins_by_sample_count(self):
        y_true = np.array([0, 1, 1, 0])
        probabilities = np.array(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.40, 0.60],
                [0.45, 0.55],
            ]
        )

        ece = top_label_ece(y_true, probabilities, n_bins=2)

        self.assertAlmostEqual(ece, 0.2125)

    def test_confidence_bins_report_empty_and_non_empty_bins(self):
        y_true = np.array([0, 1, 1, 0])
        probabilities = np.array(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.40, 0.60],
                [0.45, 0.55],
            ]
        )

        bins = build_confidence_bins(y_true, probabilities, n_bins=2)

        self.assertEqual(len(bins), 2)
        self.assertEqual(bins[0]["sample_count"], 0)
        self.assertTrue(math.isnan(bins[0]["mean_confidence"]))
        self.assertEqual(bins[1]["sample_count"], 4)
        self.assertAlmostEqual(bins[1]["mean_confidence"], 0.7125)
        self.assertAlmostEqual(bins[1]["empirical_accuracy"], 0.5)

    def test_review_triage_comparison_compares_margin_confidence_and_random(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0])
        raw_margins = np.array([0.80, 0.10, 0.70, 0.20, 0.30])
        calibrated_probabilities = np.array(
            [
                [0.90, 0.10],
                [0.55, 0.45],
                [0.20, 0.80],
                [0.48, 0.52],
                [0.40, 0.60],
            ]
        )

        rows = build_review_triage_comparison(
            y_true,
            y_pred,
            raw_margins,
            calibrated_probabilities,
            budgets=(0.40,),
        )

        by_strategy = {row["strategy"]: row for row in rows}
        self.assertEqual(by_strategy["raw_svm_margin"]["reviewed_count"], 2)
        self.assertEqual(by_strategy["raw_svm_margin"]["captured_error_count"], 2)
        self.assertEqual(
            by_strategy["calibrated_top_label_confidence"]["captured_error_count"],
            2,
        )
        self.assertAlmostEqual(
            by_strategy["random_expected"]["captured_error_count"],
            1.2,
        )

    def test_run_calibrated_confidence_analysis_on_synthetic_texts(self):
        texts = (
            ["apple banana fruit"] * 8
            + ["dog cat animal"] * 8
            + ["car bus transport"] * 8
        )
        labels = ["fruit"] * 8 + ["animal"] * 8 + ["transport"] * 8
        clean_config = {
            "split": {"test_size": 0.25, "random_state": 42, "stratify": True},
            "tfidf": {"max_features": 100, "ngram_range": [1, 2], "min_df": 1},
            "models": {
                "Linear SVM": {"dual": False, "random_state": 42},
                "Logistic Regression": {
                    "solver": "liblinear",
                    "max_iter": 200,
                    "random_state": 42,
                },
            },
        }
        calibration_config = {
            "model": {"calibration_method": "sigmoid", "cv": 2, "ensemble": True},
            "evaluation": {"confidence_bins": 5, "review_budgets": [0.5]},
        }

        analysis = run_calibrated_confidence_analysis(
            texts,
            labels,
            clean_config,
            calibration_config,
            selected_c=0.5,
        )

        self.assertEqual(len(analysis["metrics"]), 10)
        self.assertEqual(len(analysis["confidence_bins"]), 10)
        self.assertEqual(len(analysis["review_triage_curve"]), 3)
        self.assertEqual(analysis["summary"]["test_rows"], 6)

    def test_write_calibration_outputs_creates_aggregate_files_and_figures(self):
        analysis = {
            "metrics": [
                {"model": "model-a", "metric": "Log Loss", "value": 0.5, "unit": "loss"}
            ],
            "confidence_bins": [
                {
                    "model": "model-a",
                    "bin_lower": 0.0,
                    "bin_upper": 0.5,
                    "sample_count": 1,
                    "mean_confidence": 0.4,
                    "empirical_accuracy": 0.0,
                    "calibration_gap": 0.4,
                },
                {
                    "model": "model-a",
                    "bin_lower": 0.5,
                    "bin_upper": 1.0,
                    "sample_count": 1,
                    "mean_confidence": 0.8,
                    "empirical_accuracy": 1.0,
                    "calibration_gap": 0.2,
                },
            ],
            "review_triage_curve": [
                {
                    "strategy": "raw_svm_margin",
                    "budget_percent": 50.0,
                    "reviewed_count": 1,
                    "captured_error_count": 1,
                    "share_of_all_errors_captured": 1.0,
                    "errors_found_per_100_reviewed": 100.0,
                    "total_errors": 1,
                },
                {
                    "strategy": "calibrated_top_label_confidence",
                    "budget_percent": 50.0,
                    "reviewed_count": 1,
                    "captured_error_count": 1,
                    "share_of_all_errors_captured": 1.0,
                    "errors_found_per_100_reviewed": 100.0,
                    "total_errors": 1,
                },
                {
                    "strategy": "random_expected",
                    "budget_percent": 50.0,
                    "reviewed_count": 1,
                    "captured_error_count": 0.5,
                    "share_of_all_errors_captured": 0.5,
                    "errors_found_per_100_reviewed": 50.0,
                    "total_errors": 1,
                },
            ],
            "summary": {"phase": "Phase 1D", "test_rows": 2},
            "figures": {
                "y_true": np.array([0, 1]),
                "calibrated_probabilities": np.array([[0.6, 0.4], [0.2, 0.8]]),
                "logistic_probabilities": np.array([[0.7, 0.3], [0.3, 0.7]]),
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = write_calibration_outputs(
                tmp_path / "reproduced",
                tmp_path / "figures",
                "test-run",
                analysis,
            )

            self.assertTrue((run_dir / "calibration_metrics.csv").is_file())
            self.assertTrue((run_dir / "calibration_bins.csv").is_file())
            self.assertTrue((run_dir / "review_triage_calibrated_curve.csv").is_file())
            self.assertTrue((run_dir / "calibration_summary.json").is_file())
            self.assertTrue((tmp_path / "figures" / "calibration_reliability_diagram.png").is_file())
            self.assertTrue((tmp_path / "figures" / "confidence_histogram.png").is_file())
            self.assertTrue(
                (tmp_path / "figures" / "review_triage_margin_vs_calibrated_confidence.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
