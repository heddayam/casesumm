"""Build CaseSumm from opinion JSON files and extracted syllabuses."""

import argparse
import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import regex

from src.data_io import load_records

if TYPE_CHECKING:
    from tiktoken import Encoding


def flatten_list(values: list[Any]) -> Iterator[Any]:
    for value in values:
        if isinstance(value, list):
            yield from flatten_list(value)
        else:
            yield value


def trim_opinion(opinion: str) -> str:
    matches = list(regex.finditer(r"(?r)^\[.*\]", opinion, regex.BESTMATCH))
    return opinion[matches[0].end() :].strip() if matches else opinion


def make_data_json(
    encoder: "Encoding", opinion: str, syllabus: str, source: str
) -> dict[str, Any] | None:
    if not isinstance(opinion, str) or not isinstance(syllabus, str):
        raise ValueError("Opinion and syllabus must be strings")
    opinion_len = len(encoder.encode(opinion))
    if opinion_len <= 1:
        return None
    return {
        "opinion": opinion,
        "syllabus": syllabus,
        "opinion_len": opinion_len,
        "syllabus_len": len(encoder.encode(syllabus)),
        "structured_syllabus": "Held:" in syllabus,
        "source": source,
    }


def collect_records(
    encoder: "Encoding",
    pro_json: str | Path | None = None,
    syllabi: str | Path | None = None,
    super_scotus: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load opinion–syllabus pairs and count their tokens.

    encoder: Tokenizer used to calculate opinion_len and syllabus_len.
    pro_json: Directory of opinion files arranged as volume/citation.json.
    syllabi: Directory of syllabus .txt files whose names start with the citation.
        Supply this together with pro_json.
    super_scotus: File of preprocessed records with citation, opinion and syllabus.
    """
    records = []
    if pro_json is not None or syllabi is not None:
        if pro_json is None or syllabi is None:
            raise ValueError("PRO input requires both --pro-json and --syllabi")
        paths = sorted(Path(syllabi).rglob("*.txt"))
        if not paths:
            raise ValueError("No extracted syllabus .txt files found")
        for path in paths:
            citation = path.stem.split("-")[0]
            volume = citation.split(".")[0]
            if int(volume) < 14:  # Start at volume 14.
                continue
            case_path = Path(pro_json) / volume / (citation + ".json")
            if not case_path.is_file():
                raise FileNotFoundError(f"Missing opinion JSON for {citation}: {case_path}")
            majority = json.loads(case_path.read_text())["majority"]
            opinion = majority if isinstance(majority, str) else "\n".join(flatten_list(majority))
            row = make_data_json(encoder, opinion, path.read_text().strip(), "pdf-pro")
            if row is not None:
                records.append(dict(row, citation=citation))
    if super_scotus is not None:
        for record in load_records(super_scotus):
            row = make_data_json(encoder, record["opinion"], record["syllabus"], "super-scotus")
            if row is not None:
                records.append(dict(row, citation=record["citation"]))
    if not records:
        raise ValueError("No source records were produced")
    return records


def clean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove cases with missing or overly long syllabuses and reject duplicate citations."""
    cleaned = []
    seen_citations = set()
    for row in records:
        if not row["syllabus"] or row["syllabus_len"] >= row["opinion_len"]:
            continue
        citation = row["citation"]
        if citation in seen_citations:
            raise ValueError(
                f"Duplicate citation {citation}. Check for cases included in multiple sources"
            )
        seen_citations.add(citation)
        # Keep token counts from before trimming the opinion.
        cleaned.append(dict(row, opinion=trim_opinion(row["opinion"])))
    if not cleaned:
        raise ValueError("No cases remain after filtering")
    return cleaned


def read_citations(path: str | Path) -> set[str]:
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if "citation" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: expected a citation column")
        values = [row["citation"].strip() for row in reader]
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError(f"{path}: citation list must be nonempty and unique")
    return set(values)


def make_structured_splits(
    records: list[dict[str, Any]], dev_citations: set[str], test_citations: set[str]
) -> dict[str, list[dict[str, Any]]]:
    """Split cases with a "Held:" section into train, dev and test.

    dev_citations, test_citations: Sets of case citations, such as "100.US.1",
        identifying the cases in each split. Put the remaining cases in train.
    """
    if dev_citations & test_citations:
        raise ValueError("The dev and test citation lists contain some of the same cases")
    # Use the same case filters as the fine-tuning dataset loader.
    rows = [
        row
        for row in records
        if row["structured_syllabus"] == 1 and row["opinion_len"] - row["syllabus_len"] > 10
    ]
    available_citations = {row["citation"] for row in rows}
    missing_citations = (dev_citations | test_citations) - available_citations
    if missing_citations:
        raise ValueError(
            f"{len(missing_citations)} dev/test citations are missing from the dataset or were removed by filtering"
        )
    splits = {"train": [], "dev": [], "test": []}
    for row in rows:
        split = (
            "dev"
            if row["citation"] in dev_citations
            else "test"
            if row["citation"] in test_citations
            else "train"
        )
        splits[split].append(row)
    if any(not rows for rows in splits.values()):
        raise ValueError("All three experiment splits must be nonempty")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pro-json", type=Path)
    parser.add_argument("--syllabi", type=Path)
    parser.add_argument("--super-scotus", type=Path, help="Cleaned Super-SCOTUS JSON/JSONL")
    parser.add_argument("--dev-citations", type=Path)
    parser.add_argument("--test-citations", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New saved HF dataset directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists. Choose a new directory")
    if bool(args.dev_citations) != bool(args.test_citations):
        parser.error("Supply both --dev-citations and --test-citations")
    import tiktoken
    from datasets import Dataset, DatasetDict

    encoder = tiktoken.encoding_for_model("gpt-4")
    records = clean_records(
        collect_records(encoder, args.pro_json, args.syllabi, args.super_scotus)
    )
    if args.dev_citations:
        splits = make_structured_splits(
            records, read_citations(args.dev_citations), read_citations(args.test_citations)
        )
        dataset = DatasetDict({name: Dataset.from_list(rows) for name, rows in splits.items()})
        counts = {name: len(rows) for name, rows in splits.items()}
    else:
        dataset = Dataset.from_list(records)
        counts = {"corpus": len(records)}
    dataset.save_to_disk(str(args.output))
    print(f"Saved dataset to {args.output}: {counts}")


if __name__ == "__main__":
    main()
