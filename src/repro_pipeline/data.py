"""Dataset integrity validation and strict cnews parsing."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetValidationError(ValueError):
    """Raised when the local dataset fails integrity or format validation."""


@dataclass(frozen=True)
class DatasetBundle:
    texts: list[str]
    labels: list[str]
    summary: dict[str, Any]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_validated_dataset(
    path: str | Path,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
) -> DatasetBundle:
    data_path = Path(path)
    if not data_path.is_file():
        raise DatasetValidationError("Dataset file does not exist or is not a regular file.")

    size_bytes = data_path.stat().st_size
    if expected_size is not None and size_bytes != expected_size:
        raise DatasetValidationError(
            f"Dataset size mismatch: expected {expected_size}, received {size_bytes}."
        )

    sha256 = sha256_file(data_path)
    if expected_sha256 is not None and sha256.lower() != expected_sha256.lower():
        raise DatasetValidationError("Dataset SHA-256 does not match the approved file.")

    texts: list[str] = []
    labels: list[str] = []
    try:
        with data_path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\r\n")
                if not line or "\t" not in line:
                    raise DatasetValidationError(
                        f"Invalid label<TAB>text record at line {line_number}."
                    )
                label, text = line.split("\t", maxsplit=1)
                if not label or not text:
                    raise DatasetValidationError(
                        f"Empty label or text at line {line_number}."
                    )
                labels.append(label)
                texts.append(text)
    except UnicodeDecodeError as exc:
        raise DatasetValidationError("Dataset is not valid UTF-8.") from exc

    if not texts:
        raise DatasetValidationError("Dataset contains no valid records.")

    label_counts = dict(sorted(Counter(labels).items()))
    summary = {
        "size_bytes": size_bytes,
        "sha256": sha256,
        "row_count": len(texts),
        "class_count": len(label_counts),
        "label_counts": label_counts,
        "format": "label<TAB>text",
        "encoding": "utf-8",
        "invalid_row_count": 0,
    }
    return DatasetBundle(texts=texts, labels=labels, summary=summary)
