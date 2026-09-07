import re
from pathlib import Path

from bs4 import BeautifulSoup

CONCURRENCE_PATTERN = r".*Justice.*concurring.*"
DISSENT_PATTERN = r".*Justice.*, dissenting.*"


class Case:
    def __init__(self, html_content: str, filename: str) -> None:
        """Parse a case's HTML into opinion sections.

        filename: Source filename starting with volume.US.page, e.g. 100.US.1.html.
            Used to set the citation. The HTML is read from html_content.
        """
        self.section = "majority"
        self.index = 0
        self.data = {}
        self.data["majority"] = []
        self.data["concurrence"] = []
        self.data["dissent"] = []
        self.data["citation"] = ".".join(filename.split("/")[-1].split(".")[:3])
        self.process_case_text(html_content)

    def process_sections(self, paragraphs: list[str]) -> None:
        paragraphs = [p for p in paragraphs if p]
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 200:
                continue
            if re.search(CONCURRENCE_PATTERN, paragraph):
                self.data[self.section] += [paragraphs[self.index : i]]
                self.section = "concurrence"
                self.index = i
            if re.search(DISSENT_PATTERN, paragraph):
                self.data[self.section] += [paragraphs[self.index : i]]
                self.section = "dissent"
                self.index = i
        self.data[self.section] += [paragraphs[self.index :]]

    def process_case_text(self, html_content: str) -> None:
        """Remove footnotes and split the case text into opinion sections."""
        soup = BeautifulSoup(html_content, "html.parser")

        # remove footnotes and footer
        for div in soup.find_all("div", {"class": "footnote"}):
            div.decompose()
        footer = soup.find("div", id="footer")
        if footer is not None:
            footer.decompose()

        for span in soup.find_all("span", {"class": "num"}):
            span.decompose()

        text = soup.text
        opinion_pattern = r"delivered.*opinion.*[Cc]ourt.|, Circuit Justice."
        prelims = re.split(opinion_pattern, text)[0]
        body = re.split(opinion_pattern, text)[-1]

        ## Find any syllabus
        if prelims:
            omission_regex = r"Syllabus.*intentionally omitted"
            match = re.search(omission_regex, prelims)
            if not match:
                if "Syllabus" in prelims:
                    syllabus = re.split(r"[sS]yllabus", prelims)[-1]
                    end_regex = r"(^(MR\.)\sJUSTICE\s[A-Z]+)|(^[A-Z]+, J\.,?)|(\w+,\s+C\.J\.,?)"
                    self.data["syllabus"] = re.split(end_regex, syllabus)[0]

        if body:
            paragraphs = [p for p in body.split("\n") if p]
            self.process_sections(paragraphs)


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
    result = Case(args.input.read_text(), str(args.input)).data
    write_json(args.output, result, overwrite=args.overwrite)
    print(f"Saved converted case to {args.output}")


if __name__ == "__main__":
    main()
