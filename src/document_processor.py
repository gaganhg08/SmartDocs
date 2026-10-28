"""Document processing module for SmartDocs.

Handles PDF ingestion, validation, text extraction page-by-page using PyMuPDF,
empty/scanned/encrypted/corrupt document detection, and recursive text chunking
with metadata retention.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFValidationError(Exception):
    """Raised when a PDF file fails validation constraints."""
    pass


class DocumentProcessor:
    """Handles PDF validation, PyMuPDF text extraction, and text chunking."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_file_size_mb: float = 20.0,
    ) -> None:
        """Initialize DocumentProcessor configuration.

        Args:
            chunk_size: Maximum character count per document chunk (> 0).
            chunk_overlap: Overlapping character count between adjacent chunks (>= 0 and < chunk_size).
            max_file_size_mb: Maximum allowed PDF file size in megabytes.

        Raises:
            ValueError: If chunk_size or chunk_overlap parameters are invalid.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        if max_file_size_mb <= 0:
            raise ValueError(f"max_file_size_mb must be positive, got {max_file_size_mb}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size_mb = max_file_size_mb
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def validate_pdf(self, file_bytes: bytes, filename: str) -> None:
        """Validate PDF extension, file size, header, encryption, and readability.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename: Original name of the uploaded file.

        Raises:
            PDFValidationError: If the file is invalid, encrypted, corrupt, or exceeds limits.
        """
        # 1. File extension validation
        if not filename.lower().endswith(".pdf"):
            raise PDFValidationError(
                f"Invalid file format: '{filename}'. Only PDF documents (.pdf) are supported."
            )

        # 2. File size validation
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise PDFValidationError(
                f"File '{filename}' ({size_mb:.2f} MB) exceeds maximum allowed size of {self.max_file_size_mb:.1f} MB."
            )

        # 3. Empty file check
        if not file_bytes or len(file_bytes) == 0:
            raise PDFValidationError(f"File '{filename}' is empty (0 bytes).")

        # 4. PDF Magic Header check (%PDF-)
        if not file_bytes.startswith(b"%PDF-"):
            raise PDFValidationError(
                f"File '{filename}' is not a valid PDF document (missing standard %PDF header)."
            )

        # 5. PyMuPDF Readability & Encryption check
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise PDFValidationError(
                f"Failed to open '{filename}'. The file appears to be corrupted or malformed: {e}"
            )

        try:
            if doc.is_encrypted:
                # Try empty password; if still locked, fail
                if not doc.authenticate(""):
                    raise PDFValidationError(
                        f"File '{filename}' is password-protected or encrypted. Please remove encryption and re-upload."
                    )

            if len(doc) == 0:
                raise PDFValidationError(f"File '{filename}' contains 0 pages.")
        finally:
            doc.close()

    def extract_text(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Extract text page by page from PDF using PyMuPDF while preserving metadata.

        Detects scanned / image-only documents lacking extractable text.

        Args:
            file_bytes: Raw bytes of the PDF file.
            filename: Original name of the document.

        Returns:
            Dict containing:
                - filename: str
                - page_count: int
                - pages: List[Dict[str, Any]] with 1-indexed page_number and text
                - total_chars: int
                - is_scanned: bool
                - ocr_warning: Optional[str]
        """
        self.validate_pdf(file_bytes, filename)

        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        pages: List[Dict[str, Any]] = []
        total_chars = 0

        try:
            for idx, page in enumerate(doc):
                page_num = idx + 1  # 1-indexed for human readability
                raw_text = page.get_text("text") or ""
                cleaned_text = raw_text.strip()
                char_count = len(cleaned_text)
                total_chars += char_count
                pages.append({
                    "page_number": page_num,
                    "text": cleaned_text,
                    "char_count": char_count,
                })
        finally:
            doc.close()

        is_scanned = (total_chars == 0)
        ocr_warning: Optional[str] = None
        if is_scanned:
            ocr_warning = (
                f"Document '{filename}' contains {len(pages)} page(s) but no extractable text. "
                "It appears to be scanned or contains only images. "
                "Text extraction requires Optical Character Recognition (OCR), which is not currently supported in this pipeline."
            )

        return {
            "filename": filename,
            "page_count": len(pages),
            "pages": pages,
            "total_chars": total_chars,
            "is_scanned": is_scanned,
            "ocr_warning": ocr_warning,
        }

    def chunk_pages(
        self,
        pages: List[Dict[str, Any]],
        filename: str,
    ) -> List[Dict[str, Any]]:
        """Split extracted pages into overlapping text chunks, preserving source metadata.

        Args:
            pages: List of page dictionaries containing 'page_number' and 'text'.
            filename: Source document filename for attribution.

        Returns:
            List of chunk dictionaries with 'text' and 'metadata'.
        """
        chunks: List[Dict[str, Any]] = []
        chunk_counter = 0

        for page_data in pages:
            text = page_data.get("text", "")
            if not text:
                continue

            page_number = page_data.get("page_number", 1)
            split_texts = self._splitter.split_text(text)

            for split_idx, split_text in enumerate(split_texts):
                cleaned = split_text.strip()
                if not cleaned:
                    continue

                chunk_counter += 1
                chunk_id = f"{filename}_p{page_number}_c{split_idx + 1}"
                chunks.append({
                    "text": cleaned,
                    "metadata": {
                        "source": filename,
                        "page": page_number,
                        "chunk_index": split_idx + 1,
                        "chunk_id": chunk_id,
                        "char_count": len(cleaned),
                    },
                })

        return chunks

    def process_file(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """End-to-end extraction and chunking pipeline for a single PDF from bytes.

        Args:
            file_bytes: Raw bytes of the PDF file.
            filename: Source filename.

        Returns:
            Dict containing extraction details, warnings, and generated chunks.
        """
        extraction = self.extract_text(file_bytes, filename)

        if extraction["is_scanned"]:
            return {
                "filename": filename,
                "page_count": extraction["page_count"],
                "total_chars": 0,
                "is_scanned": True,
                "ocr_warning": extraction["ocr_warning"],
                "chunks": [],
                "chunk_count": 0,
                "status": "warning",
            }

        chunks = self.chunk_pages(extraction["pages"], filename)

        return {
            "filename": filename,
            "page_count": extraction["page_count"],
            "total_chars": extraction["total_chars"],
            "pages": extraction["pages"],
            "is_scanned": False,
            "ocr_warning": None,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "status": "success",
        }

    def process_file_path(self, file_path: Path | str) -> Dict[str, Any]:
        """End-to-end extraction and chunking pipeline from a filesystem path.

        Args:
            file_path: Absolute or relative path to the target PDF file.

        Returns:
            Dict containing extraction details, warnings, and generated chunks.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: '{path}'")

        file_bytes = path.read_bytes()
        return self.process_file(file_bytes, path.name)

    def load_pdf(self, file_path: Path | str) -> List[Dict[str, Any]]:
        """Load and extract pages from a PDF path preserving page metadata.

        Args:
            file_path: Path to the target PDF file.

        Returns:
            List of page dictionaries with page numbers and text.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: '{path}'")

        file_bytes = path.read_bytes()
        extraction = self.extract_text(file_bytes, path.name)
        return extraction["pages"]
