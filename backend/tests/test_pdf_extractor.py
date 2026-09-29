import pytest

from app.ingestion.pdf import PdfExtractionError, extract_pdf_pages


def test_empty_pdf_is_rejected() -> None:
    with pytest.raises(PdfExtractionError, match="empty"):
        extract_pdf_pages(b"")


def test_malformed_pdf_is_rejected() -> None:
    with pytest.raises(PdfExtractionError, match="readable PDF"):
        extract_pdf_pages(b"%PDF-1.7\nnot a real document")
