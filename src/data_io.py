"""Read and write dataset records."""

import csv
import json
from pathlib import Path
from typing import Any


def load_records(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    """Read records from JSON, JSONL, CSV or a saved Hugging Face dataset.

    path: Local file or directory created with save_to_disk, not a Hub dataset ID.
    split: Required for a saved DatasetDict. For record files, filters the
        "split" field. None reads all records.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.is_dir():
        from datasets import DatasetDict, load_from_disk

        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            if split is None:
                raise ValueError("A saved DatasetDict requires --split")
            if split not in dataset:
                raise ValueError(f"Unknown split {split!r}. Available: {list(dataset)}")
            dataset = dataset[split]
        elif split is not None:
            raise ValueError("This saved dataset has no split information. Omit --split")
        return [dict(row) for row in dataset]
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected an array of records or a JSONL/CSV file")
    if split is not None:
        if any("split" not in row for row in rows):
            raise ValueError("Input records have no split labels. Omit --split")
        rows = [row for row in rows if row["split"] == split]
    return rows


def require_fields(rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Check that each record has the required nonempty text fields."""
    if not rows:
        raise ValueError("No input records selected")
    for index, row in enumerate(rows):
        for field in fields:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Record {index}: {field} must be a nonempty string")


def write_json(path: str | Path, data: Any, overwrite: bool = False) -> None:
    path = Path(path)
    encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(encoded)
