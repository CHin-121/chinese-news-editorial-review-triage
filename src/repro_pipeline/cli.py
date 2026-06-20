"""Command-line entry point for the Phase 1B-minimal audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .classification import (
    run_clean_baseline,
    run_editorial_triage,
    run_historical_compat,
)
from .clustering import run_direct_tfidf_kmeans
from .data import load_validated_dataset
from .reporting import (
    build_comparisons,
    make_run_id,
    write_editorial_triage_outputs,
    write_outputs,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]
    return {name: importlib.metadata.version(name) for name in packages}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Local cnews.train.txt file (never copied).")
    parser.add_argument("--historical-config", default="configs/historical_compat.json")
    parser.add_argument("--clean-config", default="configs/clean_baseline.json")
    parser.add_argument("--output-root", default="results/reproduced")
    parser.add_argument(
        "--editorial-triage-only",
        action="store_true",
        help="Run only the Phase 1C-minimal clean SVM editorial-triage analysis.",
    )
    parser.add_argument(
        "--selected-c",
        type=float,
        default=0.5,
        help="Linear SVM C selected by the prior clean-baseline audit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    historical_config = _load_json(args.historical_config)
    clean_config = _load_json(args.clean_config)
    expected = historical_config["dataset"]

    dataset = load_validated_dataset(
        args.data,
        expected_size=expected["expected_size_bytes"],
        expected_sha256=expected["expected_sha256"],
    )
    print(
        f"Validated {dataset.summary['row_count']} rows across "
        f"{dataset.summary['class_count']} classes."
    )

    if args.editorial_triage_only:
        run_id = make_run_id()
        analysis = run_editorial_triage(
            dataset.texts,
            dataset.labels,
            clean_config,
            selected_c=args.selected_c,
        )
        analysis["summary"]["run_id"] = run_id
        output_directory = write_editorial_triage_outputs(
            args.output_root,
            run_id,
            analysis,
        )
        print(f"Wrote editorial-triage results to {output_directory.as_posix()}")
        return 0

    historical_result = run_historical_compat(dataset.texts, dataset.labels, historical_config)
    clean_result = run_clean_baseline(dataset.texts, dataset.labels, clean_config)
    kmeans_result = run_direct_tfidf_kmeans(dataset.texts, dataset.labels, clean_config)

    run_id = make_run_id()
    metrics = [
        *historical_result["metrics"],
        *clean_result["metrics"],
        *kmeans_result["metrics"],
    ]
    comparisons = build_comparisons(
        run_id, metrics, historical_config["historical_metrics"]
    )
    metadata = {
        "run_id": run_id,
        "phase": "Phase 1B-minimal",
        "dataset": dataset.summary,
        "git_commit": _git_commit(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "configs": {
            "historical_compat_sha256": _sha256(args.historical_config),
            "clean_baseline_sha256": _sha256(args.clean_config),
        },
        "classification": {
            "historical_compat": historical_result["details"],
            "clean_baseline": clean_result["details"],
        },
        "clustering": kmeans_result["details"],
        "scope": {
            "word2vec_run": False,
            "doc2vec_run": False,
            "svd_kmeans_run": False,
            "bert_run": False,
        },
    }
    output_directory = write_outputs(
        args.output_root, run_id, metrics, comparisons, metadata
    )
    print(f"Wrote structured results to {output_directory.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
