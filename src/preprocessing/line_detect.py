import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pdfplumber.page import Page


def detect_footer_line(page: "Page", debug: bool = False) -> np.float64 | None:
    """Return the PDF y-coordinate of the lowest horizontal line on the page."""
    # Detect horizontal lines in an image of the page.
    image = cv2.cvtColor(
        np.asarray(page.to_image(resolution=150).original.convert("RGB")), cv2.COLOR_RGB2BGR
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    detected_lines = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    contours = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    y_coords = []
    for contour in contours:
        _, y, _, _ = cv2.boundingRect(contour)
        pdf_y = (y / 150) * 72  # Assuming 150 DPI for image and 72 DPI for PDF
        y_coords.append(pdf_y)

    if debug:
        logging.getLogger(__name__).debug("Detected horizontal lines at PDF y=%s", y_coords)

    if len(y_coords) > 0:
        return np.max(y_coords)
    return None
