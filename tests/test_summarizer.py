"""Tests for document summarizer module in SmartDocs."""

import os
from unittest.mock import MagicMock, patch
import urllib.error
import pytest

from src.summarizer import DocumentSummarizer, SUMMARIZER_SYSTEM_PROMPT
from src.rag_pipeline import (
    OllamaClient,
    OllamaServerUnavailableError,
    OllamaModelNotFoundError,
)

os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class TestDocumentSummarizerEmptyInput:
    """Tests covering empty, invalid, and text-free document inputs."""

    def test_empty_document_dict(self):
        summarizer = DocumentSummarizer()
        with pytest.raises(ValueError, match="Document data is empty"):
            summarizer.summarize_document({})

    def test_scanned_document_error(self):
        summarizer = DocumentSummarizer()
        doc_data = {
            "filename": "scanned.pdf",
            "is_scanned": True,
            "pages": [{"page_number": 1, "text": ""}],
        }
        with pytest.raises(ValueError, match="appears to be scanned or contains only images"):
            summarizer.summarize_document(doc_data)

    def test_no_pages_in_document(self):
        summarizer = DocumentSummarizer()
        doc_data = {"filename": "empty_pages.pdf", "pages": []}
        with pytest.raises(ValueError, match="No pages found in document"):
            summarizer.summarize_document(doc_data)

    def test_all_pages_blank_error(self):
        summarizer = DocumentSummarizer()
        doc_data = {
            "filename": "blank.pdf",
            "pages": [
                {"page_number": 1, "text": "   "},
                {"page_number": 2, "text": "\n\n"},
            ],
        }
        with pytest.raises(ValueError, match="no extractable text"):
            summarizer.summarize_document(doc_data)


class TestDocumentSummarizerNormalInput:
    """Tests covering single-pass and multi-section summarization workflows."""

    @patch.object(OllamaClient, "is_model_available", return_value=(True, ["llama3.2:latest"]))
    @patch.object(OllamaClient, "generate")
    def test_short_document_single_pass(self, mock_generate, mock_is_avail):
        expected_summary = (
            "**Executive Overview**: Quantum computing leverages superposition and entanglement.\n"
            "**Key Points & Findings**:\n"
            "- Qubits can represent 0 and 1 simultaneously.\n"
            "- Quantum supremacy was achieved in specific benchmarks.\n"
            "**Core Conclusions**: Practical error correction remains the primary engineering hurdle."
        )
        mock_generate.return_value = expected_summary

        doc_data = {
            "filename": "quantum.pdf",
            "page_count": 2,
            "pages": [
                {"page_number": 1, "text": "Quantum computing harnesses the laws of quantum mechanics to solve problems."},
                {"page_number": 2, "text": "Fault-tolerant qubits and quantum error correction are active areas of research."},
            ],
        }

        summarizer = DocumentSummarizer(model_name="llama3.2", max_section_chars=5000)
        result = summarizer.summarize_document(doc_data)

        assert result["filename"] == "quantum.pdf"
        assert result["page_count"] == 2
        assert result["processed_pages"] == [1, 2]
        assert result["skipped_pages"] == []
        assert result["is_partial"] is False
        assert result["warning"] is None
        assert result["summary"] == expected_summary
        assert result["section_summaries"] is None
        assert result["model_used"] == "llama3.2"

        # Verify Ollama was called once with the prompt containing the pages and system prompt
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]
        assert "quantum.pdf" in call_kwargs["prompt"]
        assert "Pages 1–2" in call_kwargs["prompt"]
        assert "DOCUMENT_TEXT" in call_kwargs["prompt"]
        assert call_kwargs["system"] == SUMMARIZER_SYSTEM_PROMPT

    @patch.object(OllamaClient, "is_model_available", return_value=(True, ["llama3.2:latest"]))
    @patch.object(OllamaClient, "generate")
    def test_long_document_multi_section(self, mock_generate, mock_is_avail):
        # Mock responses: 2 section summaries and 1 synthesis summary
        mock_generate.side_effect = [
            "Section 1 Summary: Overview of renewable energy trends in 2025.",
            "Section 2 Summary: Solar photovoltaic efficiency breakthroughs and battery storage.",
            "Final Executive Summary: Renewable energy expanded significantly in 2025 led by solar efficiency and battery advances.",
        ]

        doc_data = {
            "filename": "energy_report.pdf",
            "page_count": 4,
            "pages": [
                {"page_number": 1, "text": "A" * 600},
                {"page_number": 2, "text": "B" * 600},
                {"page_number": 3, "text": "C" * 600},
                {"page_number": 4, "text": "D" * 600},
            ],
        }

        # Set max_section_chars (1500) so 4 pages of 600 chars split into exactly 2 sections of 2 pages each
        summarizer = DocumentSummarizer(model_name="llama3.2", max_section_chars=1500)
        result = summarizer.summarize_document(doc_data)

        assert result["filename"] == "energy_report.pdf"
        assert result["page_count"] == 4
        assert result["processed_pages"] == [1, 2, 3, 4]
        assert result["is_partial"] is False
        assert result["warning"] is None
        assert "Final Executive Summary" in result["summary"]
        assert result["section_summaries"] is not None
        assert len(result["section_summaries"]) >= 2

        # Verify that 3 calls were made: 2 sections + 1 synthesis
        assert mock_generate.call_count == 3


class TestDocumentSummarizerPartialAndWarning:
    """Tests covering partial processing and explicit warning messages."""

    @patch.object(OllamaClient, "is_model_available", return_value=(True, ["llama3.2:latest"]))
    @patch.object(OllamaClient, "generate")
    def test_partial_document_with_skipped_blank_page(self, mock_generate, mock_is_avail):
        mock_generate.return_value = "Summary of pages 1 and 3."

        doc_data = {
            "filename": "mixed_doc.pdf",
            "page_count": 3,
            "pages": [
                {"page_number": 1, "text": "Page one text content."},
                {"page_number": 2, "text": "   "},  # Blank page
                {"page_number": 3, "text": "Page three text content."},
            ],
        }

        summarizer = DocumentSummarizer()
        result = summarizer.summarize_document(doc_data)

        assert result["is_partial"] is True
        assert result["processed_pages"] == [1, 3]
        assert result["skipped_pages"] == [2]
        assert result["warning"] is not None
        assert "partially processed" in result["warning"]
        assert "Page(s) [2] contained no extractable text" in result["warning"]


class TestDocumentSummarizerModelUnavailable:
    """Tests covering graceful error handling when Ollama is offline or model is missing."""

    def test_ollama_offline_raises_error(self):
        # Point to an inactive port
        offline_client = OllamaClient(base_url="http://127.0.0.1:59998", timeout=1.0)
        summarizer = DocumentSummarizer(ollama_client=offline_client)

        doc_data = {
            "filename": "doc.pdf",
            "pages": [{"page_number": 1, "text": "Some text."}],
        }

        with pytest.raises(OllamaServerUnavailableError, match="Cannot connect to Ollama server"):
            summarizer.summarize_document(doc_data)

    @patch("urllib.request.urlopen")
    def test_model_missing_raises_error(self, mock_urlopen):
        # Return mock tags with a different model installed
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"models": [{"name": "gemma2:latest"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = OllamaClient()
        summarizer = DocumentSummarizer(ollama_client=client, model_name="llama3.2")

        doc_data = {
            "filename": "doc.pdf",
            "pages": [{"page_number": 1, "text": "Some text."}],
        }

        with pytest.raises(OllamaModelNotFoundError, match="Model 'llama3.2' is not installed"):
            summarizer.summarize_document(doc_data)
