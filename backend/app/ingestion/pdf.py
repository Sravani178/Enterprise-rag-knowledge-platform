from dataclasses import dataclass

import pymupdf


class PdfExtractionError(ValueError):
    """Raised when a file is not a readable PDF."""


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    page_number: int
    text: str


def extract_pdf_pages(pdf_bytes: bytes) -> list[ExtractedPage]:
    if not pdf_bytes:
        raise PdfExtractionError("The PDF is empty")

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise PdfExtractionError("The uploaded file is not a readable PDF") from exc

    try:
        if document.page_count == 0:
            raise PdfExtractionError("The PDF does not contain any pages")

        return [
            ExtractedPage(
                page_number=index + 1,
                text=document.load_page(index).get_text("text").strip(),
            )
            for index in range(document.page_count)
        ]
    finally:
        document.close()
