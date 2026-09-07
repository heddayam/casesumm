"""Convert flattened Label Studio annotations into summary rankings and error labels."""

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from src.data_io import load_records, write_json

SOURCES = {"syllabus", "oyez", "westlaw", "gpt4", "mistral"}
DIMENSIONS = {
    "relevant": "sensitivity",
    "irrelevant": "specificity",
    "clear": "clarity",
    "style": "style",
}
LETTERS = "ABCDE"


def convert_assessment(
    assessment: dict[str, Any], annotator_id: str, assessment_id: str
) -> list[dict[str, Any]]:
    """Convert one assessment into a row for each summary source.

    assessment: Flattened fields for one case, including citation and summary_order.
        summary_order lists the sources shown as A–E, in that order.
    annotator_id: Pseudonymous ID for the person who supplied the ratings.
    assessment_id: ID for this person's assessment of this case.

    Match labels A–E to source names and leave missing answers as None.
    """
    if not isinstance(assessment.get("summary_order"), str) or not isinstance(
        assessment.get("citation"), str
    ):
        raise ValueError(
            "Expected flattened annotations with citation and summary_order. Nested Label Studio exports require adaptation"
        )
    order = [source.strip() for source in assessment["summary_order"].split(",")]
    if len(order) != 5 or set(order) != SOURCES:
        raise ValueError("summary_order must contain the five distinct study sources")
    rows = {
        letter: {
            "assessment_id": assessment_id,
            "annotator_id": annotator_id,
            "citation": assessment["citation"].strip(),
            "model": "gpt4t" if source == "gpt4" else source,
        }
        for letter, source in zip(LETTERS, order)
    }
    if not assessment["citation"].strip():
        raise ValueError("Empty case citation")
    for field, name in DIMENSIONS.items():
        selected = set()
        for row in rows.values():
            row[name] = None
        for rank in range(1, 6):
            choice = assessment.get(f"{field}{rank}")
            if choice is None or choice == "":
                continue
            if choice not in rows or choice in selected:
                raise ValueError(f"{assessment_id}: invalid or duplicate summary in {field} ranks")
            rows[choice][name] = rank
            selected.add(choice)
    for position, letter in enumerate(LETTERS, 1):
        answer = assessment.get(f"error{position}")
        if answer not in {None, "", "Yes", "No"}:
            raise ValueError(f"{assessment_id}: factual-error response must be Yes, No or missing")
        rows[letter]["factual_error"] = None if answer in {None, ""} else answer == "Yes"
    return list(rows.values())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report mean ranks and factual-error counts for each summary source."""
    report = {
        "case_count": len({row["citation"] for row in rows}),
        "assessment_count": len({row["assessment_id"] for row in rows}),
        "annotator_count": len({row["annotator_id"] for row in rows}),
        "sources": {},
    }
    for source in sorted({row["model"] for row in rows}):
        source_rows = [row for row in rows if row["model"] == source]
        statistics = {}
        for name in DIMENSIONS.values():
            values = [row[name] for row in source_rows if row[name] is not None]
            statistics[name] = {
                "mean_rank": mean(values) if values else None,
                "responses": len(values),
            }
        statistics["factual_error"] = {
            "yes": sum(row["factual_error"] is True for row in source_rows),
            "no": sum(row["factual_error"] is False for row in source_rows),
            "missing": sum(row["factual_error"] is None for row in source_rows),
        }
        report["sources"][source] = statistics
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="One flattened JSON/JSONL export per annotator",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.report.resolve():
        parser.error("Data and report paths must differ")
    if args.output.exists() or args.report.exists():
        parser.error("Choose new data and report paths. Existing files are preserved")
    files = sorted(p for p in args.input_dir.iterdir() if p.suffix in {".json", ".jsonl"})
    if not files:
        parser.error("No annotation exports found")
    rows = []
    assessment_count = 0
    for annotator_index, path in enumerate(files, 1):
        for assessment in load_records(path):
            assessment_count += 1
            rows.extend(
                convert_assessment(
                    assessment, f"rater_{annotator_index:03}", f"assessment_{assessment_count:04}"
                )
            )
    if not rows:
        parser.error("No assessments found")
    write_json(args.output, rows)
    write_json(args.report, summarize(rows))
    print(f"Wrote {assessment_count} assessments ({len(rows)} summary ratings)")


if __name__ == "__main__":
    main()
