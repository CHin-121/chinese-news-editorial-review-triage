"""Structured result writing, historical comparison, and privacy checks."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_FIELDS = [
    "run_id",
    "result_type",
    "task",
    "representation",
    "model",
    "metric",
    "value",
    "unit",
    "best_params",
    "source",
]

COMPARISON_FIELDS = [
    "run_id",
    "result_type",
    "task",
    "representation",
    "model",
    "metric",
    "historical_value",
    "regenerated_value",
    "absolute_delta",
    "unit",
    "verification_status",
]


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def verification_status(historical: float, regenerated: float, unit: str) -> str:
    precision = 2 if unit == "percent" else 4
    if f"{historical:.{precision}f}" == f"{regenerated:.{precision}f}":
        return "exact_at_reported_precision"
    tolerance = 0.10 if unit == "percent" else 0.0020
    if abs(historical - regenerated) <= tolerance:
        return "near_match"
    return "not_reproduced"


def build_comparisons(
    run_id: str,
    regenerated_metrics: list[dict[str, Any]],
    historical_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    historical_index = {
        (row["task"], row["representation"], row["model"], row["metric"]): row
        for row in historical_metrics
    }
    comparisons: list[dict[str, Any]] = []
    for row in regenerated_metrics:
        key = (row["task"], row["representation"], row["model"], row["metric"])
        historical = historical_index.get(key)
        if historical is None:
            continue
        delta = abs(float(historical["value"]) - float(row["value"]))
        comparisons.append(
            {
                "run_id": run_id,
                "result_type": row["result_type"],
                "task": row["task"],
                "representation": row["representation"],
                "model": row["model"],
                "metric": row["metric"],
                "historical_value": historical["value"],
                "regenerated_value": row["value"],
                "absolute_delta": delta,
                "unit": row["unit"],
                "verification_status": verification_status(
                    float(historical["value"]), float(row["value"]), row["unit"]
                ),
            }
        )
    return comparisons


def assert_metadata_private(metadata: dict[str, Any]) -> None:
    forbidden_keys = {"text", "texts", "article", "articles", "raw_text", "news_text"}

    def inspect(value: Any, key: str | None = None) -> None:
        if key is not None:
            lowered = key.lower()
            if lowered in forbidden_keys or lowered == "path" or lowered.endswith("_path"):
                raise ValueError(f"Metadata contains forbidden field: {key}")
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                inspect(child_value, str(child_key))
        elif isinstance(value, list):
            for child in value:
                inspect(child, key)
        elif isinstance(value, str):
            if re.search(r"/Users/|/home/|[A-Za-z]:\\\\", value):
                raise ValueError("Metadata contains a private absolute path.")

    inspect(metadata)


def write_outputs(
    output_root: str | Path,
    run_id: str,
    metrics: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    assert_metadata_private(metadata)
    run_directory = Path(output_root) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    with (run_directory / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for metric in metrics:
            row = {**metric, "run_id": run_id, "best_params": json.dumps(metric["best_params"], sort_keys=True)}
            writer.writerow(row)

    with (run_directory / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparisons)

    with (run_directory / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    return run_directory
