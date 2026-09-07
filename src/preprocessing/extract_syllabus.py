import argparse
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pdfplumber

if __package__:
    from .line_detect import detect_footer_line
else:
    from line_detect import detect_footer_line
import regex
from fuzzywuzzy import fuzz

if TYPE_CHECKING:
    from pdfplumber.pdf import PDF


def unwrap_text(line: str) -> str:
    """Join a wrapped line without adding a space after a trailing hyphen."""
    if not line:
        return ""
    if line[-1] == "-":
        fixed = line[:-1]
    else:
        fixed = line + " "
    return fixed


def is_majority_upper(input_string: str) -> bool:
    """Test for a mostly uppercase heading, counting spaces as uppercase."""
    capitalized_count = sum(1 for char in input_string if char.isupper() or char == " ")
    lower_count = sum(1 for char in input_string if char.islower())
    total_letters = capitalized_count + lower_count
    if total_letters == 0:
        return False
    return (capitalized_count / total_letters) > 0.9


def identify_syllabus(line: str, line_index: int, page_num: int) -> str:
    score = fuzz.partial_ratio("Syllabus", line)
    if score > 70:
        return "syllabus"
    if line_index == 1 and not is_majority_upper(line):
        return "other"
    return ""


def check_footer_pattern(line: dict[str, Any]) -> bool:
    footer_pattern = r"^(\u2014{3}|-{3})"
    if re.search(footer_pattern, line["text"]):
        return True
    return False


def is_below_footer(line: dict[str, Any], footer_y: float) -> bool:
    """Check whether a text line is part of the footer.

    line: Text-line record from pdfplumber.
    footer_y: Distance from the top of the page to the footer, in PDF points.
    """
    if line["top"] > footer_y or check_footer_pattern(line):  # Adjust the threshold as needed
        return True
    return False


def pre108_syllabus_start(first_page_text: str) -> "list[regex.Match[str]]":
    """Find possible syllabus start points in U.S. Reports volumes before 108."""
    pattern = "(?<=\n)(.*?(APPELLANT|PLAINTIFF|DEFENDANT).*?)(?=\n)"
    fuzzy_pattern = f"({pattern}){{e<=3}}"
    matches = list(regex.finditer(fuzzy_pattern, first_page_text, regex.BESTMATCH))

    if not matches:
        pattern = r"(?s)(?<=\n)([A-Z].{1,100}?\s[vV]([a-zA-Z]?){s<=1}[[:punct:]]{1,100}.*?[\.,|A-Z])(?=\n)"
        fuzzy_pattern = f"({pattern})"  # {{e<=2}}'
        matches = list(regex.finditer(fuzzy_pattern, first_page_text, regex.BESTMATCH))

        if not matches:
            pattern = r"(?i)(?<=\n)(\s*\w+.*? v[[:punct:]][^\n]*)(?=\n)"
            matches = list(regex.finditer(pattern, first_page_text, regex.BESTMATCH))
            if not matches:
                pattern = r"(?i)(?<=\n)(.+(Claimants?){s<=3,i<=3}[[:punct:]]+)(?=\n)"
                fuzzy_pattern = f"({pattern}){{e<=1}}"
                matches = list(regex.finditer(pattern, first_page_text, regex.BESTMATCH))
    return matches


def extract_syllabus(
    pdf: "PDF", volume: int, case_name: str, debug: bool = False
) -> tuple[str, str]:
    """Extract the syllabus from a case PDF.

    volume: U.S. Reports volume number, used to select extraction rules.
    case_name: Case title, used to locate the heading on the first page.

    Returns the syllabus and the first page's text.
    """

    body = []
    footer = []
    decided = False
    if not pdf.pages:
        raise ValueError("PDF has no pages")
    first_page_text = pdf.pages[0].extract_text(x_tolerance=1) or ""
    if not first_page_text.strip():
        return "", first_page_text

    patterns_to_exclude = [r"Opinions Per Curiam"]
    for pattern in patterns_to_exclude:
        if re.search(pattern, first_page_text):
            return "", first_page_text  # str(pattern)

    if volume < 108:  # 112 is where decided officially doesnt become reliable, ot sure about 108
        matches = pre108_syllabus_start(first_page_text)
    else:
        pattern = r"((Decided){e<=4}\s([^ ]{3,9}\s?[A-Za-z0-9]+(th|st|rd)?[[:punct:]]?\s?(\d\d){e<=2}).*?)(?=\n)"
        matches = list(regex.finditer(pattern, first_page_text, regex.BESTMATCH))
        if not matches or "U. S." in matches[0][0] or "U.S." in matches[0][0]:
            logging.getLogger(__name__).debug("No decision-date heading found on the first page")
            return "", first_page_text
        else:
            decided = True

    span_end = -1

    if matches:
        if volume < 108:
            match = matches[-1]
            line_upper_ratio = sum(1 for c in match[0] if c.isupper()) / len(match[0])
            for candidate in matches:
                current_upper_ratio = sum(1 for c in candidate[0] if c.isupper()) / len(
                    candidate[0]
                )
                if current_upper_ratio > line_upper_ratio and line_upper_ratio < 0.25:
                    line_upper_ratio = current_upper_ratio
                    match = candidate

            span_end = match.span()[1]
        else:
            match = matches[0]
            span_end = match.span()[1]

    if volume < 109 and not decided:
        if matches:
            initial_upper_ratio = sum(1 for c in match[0] if c.isupper()) / len(match[0])
            case_name_score = fuzz.partial_token_set_ratio(case_name, match[0])
        else:
            initial_upper_ratio = 0
            case_name_score = 0
        if initial_upper_ratio < 0.2 or len(matches) > 1:
            char_offset = 0
            for i, line in enumerate(first_page_text.split("\n")):
                char_offset += len(line) + 1
                if volume < 109:
                    if not line.strip() or line.startswith("VOL."):
                        continue
                    current_upper_ratio = sum(1 for c in line if c.isupper()) / len(line)
                    current_case_name_score = fuzz.partial_token_set_ratio(case_name, line)
                    if (
                        current_upper_ratio > 0.2
                        and current_case_name_score > case_name_score * 1.2
                    ):  # current_upper_ratio > initial_upper_ratio and
                        case_name_score = current_case_name_score
                        span_end = char_offset - 1

    for line in first_page_text[span_end + 1 :].split("\n"):
        words = line.split()
        if len(line) == 0:
            continue
        current_upper_ratio = sum(1 for c in line if c.isupper()) / len(line)
        if current_upper_ratio > 0.2 or len(words) < 3:
            span_end += len(line) + 1
        else:
            break
    left_newline = first_page_text[span_end + 1 :].find("\n")
    if left_newline > 3:
        left_newline = 0
    for line in first_page_text[span_end + 1 :][left_newline + 1 :].split("\n"):
        n_words = len(line.split())
        if n_words == 0:
            n_words = 1
        hyphens_frac = sum(1 for c in line if c == "-") / n_words
        if hyphens_frac > 0.25 or n_words < 5:
            logging.getLogger(__name__).debug("Skipping heading line: %s", line)
            span_end += len(line) + 1
            left_newline = first_page_text[span_end:].find("\n")
            if left_newline < 5:
                span_end += left_newline
        else:
            break

    if span_end == -1:
        return "", first_page_text

    char_offset = 0
    end_line = None

    for i, line in enumerate(first_page_text.split("\n")):
        char_offset += len(line) + 1
        if span_end <= char_offset - 1:
            end_line = i
            break
    if end_line is None:
        raise ValueError("Could not locate the syllabus start line")
    syllabus_start_idx = end_line + 1

    is_body = True
    char_widths = []
    font_sizes = []
    new_page_indices = []
    prev_page_section = -1
    for page_num, page in enumerate(pdf.pages):
        is_footer = False
        current_body = []
        current_footer = []
        current_char_widths = []
        current_font_sizes = []
        body_y = 10000  # bigger than largest y coordinate on page
        footer_y = detect_footer_line(page, debug=debug)
        lines = page.extract_text_lines(
            return_chars=True, strip=False, layout=False, x_tolerance=1, x_density=4, y_density=11
        )
        if len(lines) > 0 and "[" in lines[0]["text"]:
            lines = lines[1:]
            syllabus_start_idx -= 1
        section_name = "" if page_num != 0 else "syllabus"
        has_vertical_gap = False
        for i, line in enumerate(lines):
            if len(line["text"].strip()) < 2:
                continue
            if page_num == 0:
                if i < syllabus_start_idx:
                    continue
                elif i == syllabus_start_idx:
                    body_y = line["top"]
            text = unwrap_text(line["text"])
            if section_name == "":
                section_name = identify_syllabus(text, i, page_num)
            elif section_name != "":  # or prev_page_section == num_page-1:
                if is_footer or (
                    footer_y and footer_y > body_y and is_below_footer(line, footer_y)
                ):
                    current_footer.append(text)
                    is_footer = True
                elif is_body:
                    syllabus_end_pattern = r"(^(MR\.)(\sCHIEF)?\sJUSTICE\s[A-Z]+)|(^[A-Z]+, J\.,?)|(\w+,\s+C\.J\.,?)|(^APPEAL)|(^ERROR)|(^IN error)|(^THIS)"
                    if re.match(syllabus_end_pattern, text):
                        section_name = ""
                        break
                    current_body.append(text)
                    current_font_sizes.append(
                        np.median([round(char["size"]) for char in line["chars"]])
                    )
                    if len(text) > 0:
                        current_char_widths.append(2 * (line["x1"] - line["x0"]) / len(text))
                    if (prev_page_section < page_num) and i + 1 < len(lines):
                        current_line_bottom = line["bottom"]
                        next_line_top = lines[i + 1]["top"]
                        space_to_next_line = next_line_top - current_line_bottom
                        current_line_height = line["bottom"] - line["top"]
                        has_vertical_gap = round(space_to_next_line, 1) > round(
                            current_line_height - (current_line_height * 0.3), 1
                        )
                        if has_vertical_gap:
                            if (
                                footer_y
                                and footer_y > body_y
                                and is_below_footer(lines[i + 1], footer_y)
                            ):
                                is_footer = True
                            else:
                                section_name = ""
                                break
        if volume < 70:
            if (
                current_body
                and body
                and current_font_sizes
                and font_sizes
                and np.median(current_font_sizes) > np.median(font_sizes)
                and (
                    not current_body[0].lower()
                    and not body[-1].islower()
                    and np.median(current_font_sizes) - np.median(font_sizes) <= 1
                )
            ):
                break
            else:
                body += current_body
                char_widths += current_char_widths
                font_sizes += current_font_sizes
                footer += current_footer
                prev_page_section = page_num
                new_page_indices.append(len(body))
                if has_vertical_gap:
                    break
        else:
            body += current_body
            char_widths += current_char_widths
            font_sizes += current_font_sizes
            footer += current_footer
            if section_name == "syllabus":
                prev_page_section = page_num
                new_page_indices.append(len(body))
            elif prev_page_section < page_num:  # section_name == "":# and i > 3:
                break

    if new_page_indices and len(char_widths) > new_page_indices[-1]:  # and not end:
        similar_char_widths = (
            np.abs(
                np.array(char_widths[new_page_indices[-1] :]).round()
                - np.mean(char_widths[: new_page_indices[-1]]).round()
            )
            <= 1
        )
        syllabus_end_idx = np.argmax(np.logical_not(similar_char_widths))

        if np.logical_not(similar_char_widths).sum() > 1:  # syllabus_end_idx > 0 and
            body = body[: syllabus_end_idx + new_page_indices[-1]]

    text = "".join(body)
    return text, first_page_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a syllabus from one source PDF")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--volume", "--vol", type=int, required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists. Choose a new path or pass --overwrite")
    with pdfplumber.open(args.pdf) as pdf:
        syllabus, _ = extract_syllabus(pdf, args.volume, args.case_name, args.debug)
    if not syllabus.strip():
        parser.error("No syllabus found. Source requires inspection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
        stream.write(syllabus)
    print(f"Saved syllabus to {args.output}")


if __name__ == "__main__":
    main()
