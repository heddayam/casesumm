import logging
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import DATA_DIR

# Directory for pdfs
DIR = str(DATA_DIR / "cases")
BASE_DIR = str(DATA_DIR) + "/"

# Setup logger
logger = logging.getLogger(__name__)


def fetch(url: str) -> requests.Response:
    """Fail visibly on HTTP errors and retain the historical request interval."""
    response = requests.get(url, timeout=60)
    time.sleep(7)
    response.raise_for_status()
    return response


def save_pdf(url: str, year: int | str, name: str) -> None:
    response = fetch(url)
    if not response.content.lstrip().startswith(b"%PDF-"):
        raise ValueError(f"Expected a PDF response for {name}")
    if not re.fullmatch(r"[0-9]+|nan", str(year)):
        raise ValueError("Expected a numeric year or nan")
    filename = re.sub(r"[/\\]", "_", name).strip()[:50]
    if not filename:
        raise ValueError("Case name must be nonempty")
    pdf_path = Path(DIR) / str(year) / f"{filename}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    # Truncated case names can collide; never silently replace an earlier download.
    with pdf_path.open("xb") as pdf_file:
        pdf_file.write(response.content)
    logger.info("Saved %s", pdf_path)


def justia_scraper() -> None:
    """Download Justia case PDFs from 2008 through 2014."""
    url = "https://supreme.justia.com/cases/federal/us/year/{}.html"

    for year in range(2008, 2015):
        response = fetch(url.format(year))

        if response.status_code == 200:
            year_page = BeautifulSoup(response.text, "html.parser")

            cases = year_page.find_all(class_="has-padding-content-block-30 -zb search-result")

            for case in cases:
                print("-------------------------------------")
                name = case.find("a", class_="case-name").get_text().strip()
                name = name.replace("/", "_")
                case_url = (
                    case.find(class_="color-green text-soft-wrap").find("a").get_text().strip()
                )
                response = fetch(case_url)
                case_page = BeautifulSoup(response.text, "html.parser")
                try:
                    pdf_url = (
                        case_page.find(class_="buttons-list-item").find_all("a")[0].get("href")
                    )
                except (AttributeError, IndexError):
                    logger.warning("No PDF link found for %s", case_url)
                    continue
                print(pdf_url)

                save_pdf(pdf_url, year, name)


def loc_case_name() -> None:
    """Collect case names and citations from Library of Congress volumes."""
    case_citations = []
    volume_template = "https://www.loc.gov/collections/united-states-reports/?fa=partof%3Au.s.+reports%3A+volume+{}&st=list&c=150"

    for volume_num in tqdm(range(1, 515)):
        formatted_num = str(volume_num).zfill(3)
        volume_url = volume_template.format(formatted_num)
        response = fetch(volume_url)

        if response.status_code == 200:
            volume_soup = BeautifulSoup(response.text, "html.parser")
            cases = volume_soup.find(id="results").find_all(class_="item-description-title")

            for case in cases:
                case_text = case.find("a").text
                name = (
                    case_text.split("U.S. Reports: ")[-1]
                    .split(f", {volume_num}")[0]
                    .replace("/", "\\")
                )
                pattern = r"\(\d{4}\)\."
                citation = str(volume_num) + case_text.split(f", {volume_num}")[-1]
                cleaned_citation = re.sub(pattern, "", citation).replace("\n", "").replace("\r", "")
                case_citations.append([name, cleaned_citation])

    with open(os.path.join(BASE_DIR, "case-cite-pairs.txt"), "x", encoding="utf-8") as output_file:
        for pair in case_citations:
            line = "\t".join(pair)
            output_file.write(line + "\n")


def loc_scraper() -> None:
    """Download Library of Congress PDFs for volumes 243–499."""
    volume_template = "https://www.loc.gov/collections/united-states-reports/?fa=partof%3Au.s.+reports%3A+volume+{}&st=list&c=150"

    for volume_num in range(243, 500):
        formatted_num = str(volume_num).zfill(3)
        volume_url = volume_template.format(formatted_num)
        response = fetch(volume_url)

        if response.status_code == 200:
            volume_soup = BeautifulSoup(response.text, "html.parser")
            cases = volume_soup.find(id="results").find_all(class_="item-description-title")

            for case in cases:
                case_url = case.find("a").get("href")
                response = fetch(case_url)

                if response.status_code == 200:
                    case_soup = BeautifulSoup(response.text, "html.parser")
                    pdf_url = (
                        case_soup.find(class_="select-default")
                        .select('option[data-file-download="PDF"]')[0]
                        .get("value")
                    )
                    citation = case_soup.find(class_="item-title").find_all("cite")[0].get_text()
                    name = (
                        citation.split("U.S. Reports: ")[-1]
                        .split(f", {volume_num}")[0]
                        .replace("/", "\\")
                    )

                    match = re.search(r"\(\d{4}\)", citation)
                    if match:
                        year = match.group().strip("(").strip(")")
                    else:
                        year = "nan"

                    save_pdf(pdf_url, year, name)


def main() -> None:
    import argparse
    from pathlib import Path

    global DIR, BASE_DIR
    parser = argparse.ArgumentParser(
        description="Download case PDFs or collect case names and citations."
    )
    parser.add_argument(
        "--source", choices=["loc-citations", "loc-pdfs", "justia-pdfs"], required=True
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for downloaded files"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory to preserve existing downloads")
    args.output_dir.mkdir(parents=True)
    DIR = BASE_DIR = str(args.output_dir.resolve())
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    {"loc-citations": loc_case_name, "loc-pdfs": loc_scraper, "justia-pdfs": justia_scraper}[
        args.source
    ]()


if __name__ == "__main__":
    main()
