"""Tests for document processor module in SmartDocs."""

import pytest
import pymupdf
from src.document_processor import DocumentProcessor, PDFValidationError


def create_test_pdf(pages_text: list[str]) -> bytes:
    """Helper utility to generate in-memory PDF byte content with PyMuPDF."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text.strip():
            page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestDocumentProcessorValidation:
    """Tests covering file validation constraints."""

    def test_invalid_extension(self):
        processor = DocumentProcessor()
        with pytest.raises(PDFValidationError, match="Invalid file format"):
            processor.validate_pdf(b"%PDF-1.4 dummy", "report.docx")

    def test_empty_file_bytes(self):
        processor = DocumentProcessor()
        with pytest.raises(PDFValidationError, match="is empty"):
            processor.validate_pdf(b"", "empty.pdf")

    def test_invalid_pdf_header(self):
        processor = DocumentProcessor()
        with pytest.raises(PDFValidationError, match="missing standard %PDF header"):
            processor.validate_pdf(b"NOT_A_PDF_FILE", "corrupt.pdf")

    def test_file_size_limit_exceeded(self):
        processor = DocumentProcessor(max_file_size_mb=0.001)  # ~1 KB limit
        large_bytes = b"%PDF-1.4 " + b"X" * 2048
        with pytest.raises(PDFValidationError, match="exceeds maximum allowed size"):
            processor.validate_pdf(large_bytes, "big.pdf")

    def test_corrupted_pdf_structure(self):
        processor = DocumentProcessor()
        # Has %PDF- header but malformed trailer/body
        corrupt_bytes = b"%PDF-1.5 \n corrupted body content without valid xref table"
        with pytest.raises(PDFValidationError, match="corrupted or malformed"):
            processor.validate_pdf(corrupt_bytes, "broken.pdf")

    def test_invalid_chunk_parameters(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            DocumentProcessor(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            DocumentProcessor(chunk_size=500, chunk_overlap=-10)

        with pytest.raises(ValueError, match="strictly less than chunk_size"):
            DocumentProcessor(chunk_size=500, chunk_overlap=500)


class TestTextExtraction:
    """Tests covering PyMuPDF text extraction and scanned PDF detection."""

    def test_extract_text_multi_page(self):
        processor = DocumentProcessor()
        pdf_bytes = create_test_pdf([
            "Welcome to the first page of SmartDocs documentation.",
            "This is the second page explaining vector search and FAISS.",
        ])

        result = processor.extract_text(pdf_bytes, "manual.pdf")

        assert result["filename"] == "manual.pdf"
        assert result["page_count"] == 2
        assert result["is_scanned"] is False
        assert result["ocr_warning"] is None
        assert len(result["pages"]) == 2
        assert result["pages"][0]["page_number"] == 1
        assert "first page" in result["pages"][0]["text"]
        assert result["pages"][1]["page_number"] == 2
        assert "second page" in result["pages"][1]["text"]

    def test_scanned_pdf_detection(self):
        processor = DocumentProcessor()
        # Create a PDF with blank pages (simulating scanned images without OCR layer)
        scanned_bytes = create_test_pdf(["", "   \n  "])

        result = processor.extract_text(scanned_bytes, "scanned_doc.pdf")

        assert result["is_scanned"] is True
        assert result["total_chars"] == 0
        assert result["ocr_warning"] is not None
        assert "Optical Character Recognition (OCR)" in result["ocr_warning"]
        assert "not currently supported" in result["ocr_warning"]


class TestTextChunking:
    """Tests covering chunking logic and metadata attribution."""

    def test_chunk_pages_preserves_metadata(self):
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        pages = [
            {"page_number": 1, "text": "A" * 150},
            {"page_number": 2, "text": "B" * 80},
        ]

        chunks = processor.chunk_pages(pages, "sample.pdf")

        assert len(chunks) >= 2
        # Verify first chunk metadata
        assert chunks[0]["metadata"]["source"] == "sample.pdf"
        assert chunks[0]["metadata"]["page"] == 1
        assert chunks[0]["metadata"]["chunk_index"] == 1
        assert chunks[0]["metadata"]["chunk_id"] == "sample.pdf_p1_c1"

        # Verify second page chunk metadata
        page_2_chunks = [c for c in chunks if c["metadata"]["page"] == 2]
        assert len(page_2_chunks) >= 1
        assert page_2_chunks[0]["metadata"]["source"] == "sample.pdf"

    def test_process_file_end_to_end(self):
        processor = DocumentProcessor(chunk_size=150, chunk_overlap=30)
        pdf_bytes = create_test_pdf([
            "Artificial Intelligence enables modern software applications to reason.",
            "Sentence transformers encode dense vector representations for FAISS indexing.",
        ])

        res = processor.process_file(pdf_bytes, "ai_overview.pdf")

        assert res["status"] == "success"
        assert res["page_count"] == 2
        assert res["chunk_count"] >= 2
        assert res["is_scanned"] is False
        assert len(res["chunks"]) == res["chunk_count"]
