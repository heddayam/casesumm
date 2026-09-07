import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

CONCURRENCE_PATTERN = r".*Justice.*concurring.*"
DISSENT_PATTERN = r".*Justice.*dissenting.*"


def add_section(section: str, paragraphs: list[str], data: dict[str, Any]) -> dict[str, Any]:
    if section not in data:
        data[section] = []
    data[section].append("\n".join(paragraphs))

    return data


def process_html(html_content: str) -> dict[str, Any]:
    """Extract case metadata and group the numbered opinion paragraphs."""
    data = {}
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract citation
    if soup.title:
        data["citation"] = soup.title.text

    parties = soup.find("p", class_="parties")
    if parties:
        data["case_name"] = parties.text

    # Extract prelims, which normally contain the syllabus
    prelims = soup.find("div", class_="prelims")
    if prelims:
        if "Syllabus" in prelims.text:
            data["syllabus"] = prelims.text

    body = soup.find_all("div", class_="num")
    if body:
        paragraphs = [" ".join(p.text.split("\n")[2:]) for p in body]
        section = "majority"
        section_start = 0
        data["majority"] = []
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 200:
                continue
            if re.search(CONCURRENCE_PATTERN, paragraph):
                data = add_section(section, paragraphs[section_start:i], data)
                section = "concurrence"
                section_start = i
            if re.search(DISSENT_PATTERN, paragraph):
                data = add_section(section, paragraphs[section_start:i], data)
                section = "dissent"
                section_start = i
        data = add_section(section, paragraphs[section_start:], data)

    return data


def main() -> None:
    import argparse

    from src.data_io import write_json

    parser = argparse.ArgumentParser(description="Convert a PRO case HTML file to JSON")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists. Choose a new path or pass --overwrite")
    result = process_html(args.input.read_text())
    write_json(args.output, result, overwrite=args.overwrite)
    print(f"Saved converted case to {args.output}")


if __name__ == "__main__":
    main()
