from typing import Any

import regex


def identify_opinion(justia_sections: dict[str, str]) -> str | None:
    for section, text in justia_sections.items():
        if "Opinion" in section:
            return text


def trim_opinion(opinion: str) -> str | None:
    """Remove the opinion heading, returning None when no heading matches."""
    pattern = r"(((delivered \w+ opinion \w+ \w+ [Cc]ourt\..*?)|(, Circuit Justice.?)|(per curiam.?)|([Cc]ourt \w+ \w+ delivered \w+ opinion .*?))(\n|\s*))"
    fuzzy_pattern = f"({pattern}){{e<=3}}"
    match = regex.search(fuzzy_pattern, opinion, regex.BESTMATCH)
    if match:
        span_end = match.span()[1]
    else:
        return None

    trimmed = opinion[span_end:].strip()
    trimmed = regex.sub(r"[[:blank:]]+", " ", trimmed)
    return trimmed


def trim_syllabus(syllabus: str) -> str | None:
    """Keep syllabus text after the last matching decision-date heading."""
    pattern = r"(Decided (\w+ \d+, \d+).*\n)"
    fuzzy_pattern = f"({pattern}){{e<=3}}"
    matches = list(regex.finditer(fuzzy_pattern, syllabus, regex.BESTMATCH))
    matches.reverse()
    if matches:
        span_end = matches[0].span()[1]
    else:
        return None
    trimmed = syllabus[span_end:].strip()
    trimmed = regex.sub(r"[[:blank:]]+", " ", trimmed)
    return trimmed


def main() -> None:
    """Preprocess Super-SCOTUS opinions and syllabuses for dataset construction."""
    import argparse
    from pathlib import Path

    from src.data_io import load_records, write_json

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Super-SCOTUS records in JSON or JSONL format"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists. Choose a new path or pass --overwrite")
    rows = preprocess_records(load_records(args.input))
    if not rows:
        parser.error("No opinion–syllabus pairs could be extracted from the input")
    write_json(args.output, rows, overwrite=args.overwrite)
    print(f"Saved {len(rows)} opinion–syllabus pairs to {args.output}")


def preprocess_records(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract opinion–syllabus pairs and format citations as volume.US.page."""
    rows = []
    for index, record in enumerate(records):
        citation = record.get("citation")
        sections = record.get("justia_sections")
        if not isinstance(citation, str) or not isinstance(sections, dict):
            raise ValueError(f"Record {index}: citation and justia_sections are required")
        opinion = identify_opinion(sections)
        syllabus = sections.get("Syllabus")
        if not isinstance(opinion, str) or not isinstance(syllabus, str):
            continue
        opinion, syllabus = trim_opinion(opinion), trim_syllabus(syllabus)
        if not opinion or not syllabus:
            continue
        citation = citation.replace(" ", ".")
        if "_" in citation:
            if not record.get("id"):
                raise ValueError(f"Record {index}: incomplete citation requires an id")
            citation = citation.replace("_", str(record["id"]))
        rows.append(
            {
                "citation": citation,
                "opinion": opinion,
                "syllabus": syllabus,
                "source": "super-scotus",
            }
        )
    return rows


if __name__ == "__main__":
    main()
