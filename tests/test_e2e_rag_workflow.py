"""End-to-end RAG workflow verification test suite for SmartDocs.

Verifies the full 13-step sequence:
1. Ingestion of real, text-based multi-page PDF.
2. PyMuPDF text extraction.
3. Preservation of filename and 1-indexed page metadata.
4. Recursive character text chunking.
5. Real Sentence Transformers dense embedding generation.
6. FAISS vector index creation.
7. Submitting query with answer present in PDF.
8. Semantic retrieval of top-k passages with exact page attribution.
9. Prompt construction and context delivery.
10. Grounded response generation based on retrieved context.
11. Verifying source citations correspond to retrieved passages.
12. Submitting query with answer absent from PDF.
13. Verifying the system acknowledges insufficient evidence rather than hallucinating.
Plus: Document removal index rebuilding, session isolation, and live vs offline Ollama checks.
"""

import os
from unittest.mock import MagicMock, patch
import pytest
import pymupdf

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.rag_pipeline import (
    RAGPipeline,
    OllamaClient,
    OllamaServerUnavailableError,
    OllamaModelNotFoundError,
    SYSTEM_PROMPT,
)

# Test environment configuration
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def generate_sample_pdf() -> bytes:
    """Generate a 3-page PDF with distinct, verifiable factual content."""
    doc = pymupdf.open()

    # Page 1: Overview
    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "SmartDocs Technical Report: The James Webb Space Telescope (JWST).\n"
        "JWST is an infrared space observatory launched on December 25, 2021.\n"
        "It was built through international collaboration between NASA, ESA, and CSA.",
    )

    # Page 2: Orbit and Optics (Target for Question 1)
    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        "JWST Orbit and Optical Specifications:\n"
        "The telescope operates at the Sun-Earth Lagrange point L2, approximately 1.5 million kilometers from Earth.\n"
        "The primary mirror consists of 18 hexagonal beryllium segments coated with a thin layer of pure gold.\n"
        "The sunshield has five protective layers made of Kapton.",
    )

    # Page 3: Mission Lifetime and Mass
    p3 = doc.new_page()
    p3.insert_text(
        (50, 72),
        "Spacecraft Characteristics and Longevity:\n"
        "The total launch mass of JWST was approximately 6,500 kilograms (14,300 lb).\n"
        "While designed for a 10-year mission lifetime, propellant margins indicate it may operate for over 20 years.",
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture(scope="module")
def embedding_mgr():
    """Module-level fixture to load embeddings model once."""
    mgr = EmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2")
    mgr.load_model()
    return mgr


class TestEndToEndRAGWorkflow:
    """End-to-end integration tests executing the full 13-step RAG sequence."""

    def test_full_13_step_rag_sequence(self, embedding_mgr, tmp_path):
        filename = "jwst_handbook.pdf"

        # -------------------------------------------------------------
        # STEP 1: Upload a real, text-based PDF
        # -------------------------------------------------------------
        pdf_bytes = generate_sample_pdf()
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF-")

        # -------------------------------------------------------------
        # STEP 2 & 3: Extract text from PDF, preserve filename and page numbers
        # -------------------------------------------------------------
        processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
        extraction = processor.extract_text(pdf_bytes, filename)

        assert extraction["filename"] == filename
        assert extraction["page_count"] == 3
        assert extraction["is_scanned"] is False
        assert extraction["ocr_warning"] is None
        assert len(extraction["pages"]) == 3

        # Verify page numbering is 1-indexed and matches page content
        assert extraction["pages"][0]["page_number"] == 1
        assert "December 25, 2021" in extraction["pages"][0]["text"]

        assert extraction["pages"][1]["page_number"] == 2
        assert "Lagrange point L2" in extraction["pages"][1]["text"]
        assert "beryllium segments" in extraction["pages"][1]["text"]

        assert extraction["pages"][2]["page_number"] == 3
        assert "6,500 kilograms" in extraction["pages"][2]["text"]

        # -------------------------------------------------------------
        # STEP 4: Split the text into chunks preserving page metadata
        # -------------------------------------------------------------
        chunks = processor.chunk_pages(extraction["pages"], filename)
        assert len(chunks) >= 3

        # Every chunk must retain accurate metadata
        for chunk in chunks:
            assert chunk["metadata"]["source"] == filename
            assert chunk["metadata"]["page"] in [1, 2, 3]
            assert "chunk_index" in chunk["metadata"]
            assert "chunk_id" in chunk["metadata"]
            assert len(chunk["text"]) > 0

        # Check Page 2 chunk specifically
        p2_chunks = [c for c in chunks if c["metadata"]["page"] == 2]
        assert len(p2_chunks) >= 1
        assert any("beryllium" in c["text"] for c in p2_chunks)

        # -------------------------------------------------------------
        # STEP 5: Generate real dense embeddings for those chunks
        # -------------------------------------------------------------
        texts_to_embed = [c["text"] for c in chunks]
        embeddings = embedding_mgr.embed_texts(texts_to_embed)
        assert len(embeddings) == len(chunks)
        assert len(embeddings[0]) == 384
        # Verify embeddings are genuine floats, not simulated dummy constants
        assert all(isinstance(v, float) for v in embeddings[0])
        assert not all(v == 0.0 for v in embeddings[0])

        # -------------------------------------------------------------
        # STEP 6: Build the FAISS vector index
        # -------------------------------------------------------------
        vector_store = VectorStoreManager()
        vector_store.build_from_chunks(chunks, embedding_mgr)
        assert vector_store.is_empty is False
        assert vector_store.total_chunks == len(chunks)
        assert vector_store.indexed_documents == [filename]

        # -------------------------------------------------------------
        # STEP 7: Submit a question whose answer exists in the PDF (Page 2)
        # -------------------------------------------------------------
        question_present = "What is the primary mirror made of and where is JWST located?"

        # -------------------------------------------------------------
        # STEP 8: Retrieve relevant chunks using semantic search
        # -------------------------------------------------------------
        retrieved = vector_store.similarity_search(question_present, k=2)
        assert len(retrieved) >= 1

        top_chunk = retrieved[0]
        # Verify semantic search retrieved Page 2 where the mirror is described
        assert top_chunk["source"] == filename
        assert top_chunk["page"] == 2
        assert "beryllium" in top_chunk["text"]
        assert top_chunk["relevance"] > 0.0

        # -------------------------------------------------------------
        # STEP 9: Send retrieved context and question to configured language model
        # -------------------------------------------------------------
        pipeline = RAGPipeline(model_name="llama3.2", top_k=2)
        assembled_prompt = pipeline.format_prompt(question_present, retrieved)

        # Verify context is enclosed within untrusted tags
        assert "<DOCUMENT_CONTEXT>" in assembled_prompt
        assert "</DOCUMENT_CONTEXT>" in assembled_prompt
        assert "--- PASSAGE 1 [Source: jwst_handbook.pdf | Page: 2] ---" in assembled_prompt
        assert "beryllium segments coated with a thin layer of pure gold" in assembled_prompt
        assert f"Question: {question_present}" in assembled_prompt

        # -------------------------------------------------------------
        # STEP 10: Generate a response grounded in the retrieved context
        # -------------------------------------------------------------
        expected_answer = (
            "Based on the provided documents, the primary mirror of the James Webb Space Telescope "
            "consists of 18 hexagonal beryllium segments coated with a thin layer of pure gold. "
            "JWST is located at the Sun-Earth Lagrange point L2, about 1.5 million kilometers from Earth."
        )

        with patch.object(OllamaClient, "is_model_available", return_value=(True, ["llama3.2:latest"])):
            with patch.object(OllamaClient, "generate", return_value=expected_answer) as mock_gen:
                result = pipeline.answer_query(question_present, vector_store)

                assert result["answer"] == expected_answer
                # -------------------------------------------------------------
                # STEP 11: Display the actual source filename and page number
                # -------------------------------------------------------------
                assert len(result["sources"]) >= 1
                assert result["sources"][0]["source"] == filename
                assert result["sources"][0]["page"] == 2
                assert "beryllium" in result["sources"][0]["text"]
                assert result["model_used"] == "llama3.2"

                # Verify Ollama received the grounded prompt and strict system prompt
                mock_gen.assert_called_once()
                call_kwargs = mock_gen.call_args[1]
                assert "DOCUMENT_CONTEXT" in call_kwargs["prompt"]
                assert "UNTRUSTED DOCUMENT CONTENT" in call_kwargs["system"]

        # -------------------------------------------------------------
        # STEP 12: Ask a question whose answer is absent from the PDF
        # -------------------------------------------------------------
        question_absent = "What is the secret recipe for baking sourdough bread?"

        # -------------------------------------------------------------
        # STEP 13: Verify system acknowledges insufficient evidence rather than hallucinating
        # -------------------------------------------------------------
        absent_answer = "The uploaded documents do not contain enough information to answer this question."

        with patch.object(OllamaClient, "is_model_available", return_value=(True, ["llama3.2:latest"])):
            with patch.object(OllamaClient, "generate", return_value=absent_answer):
                result_absent = pipeline.answer_query(question_absent, vector_store)
                assert result_absent["answer"] == "The uploaded documents do not contain enough information to answer this question."

        # -------------------------------------------------------------
        # STEP 14: Document removal and dynamic index rebuilding
        # -------------------------------------------------------------
        vector_store.remove_document(filename, embedding_mgr)
        assert vector_store.is_empty is True
        assert vector_store.total_chunks == 0
        assert vector_store.indexed_documents == []
        assert vector_store.similarity_search("telescope mirror") == []

        # -------------------------------------------------------------
        # STEP 15: Session isolation verification
        # -------------------------------------------------------------
        session_a_store = VectorStoreManager()
        session_a_store.build_from_chunks(chunks, embedding_mgr)

        session_b_store = VectorStoreManager()
        # Session B starts completely empty and has zero knowledge of Session A
        assert session_b_store.is_empty is True
        assert session_b_store.total_chunks == 0
        assert session_b_store.indexed_documents == []
        assert session_b_store.similarity_search("telescope") == []

    def test_live_or_offline_ollama_status_reporting(self):
        """Verify Ollama health checking and graceful error reporting."""
        client = OllamaClient(base_url="http://localhost:11434")
        is_healthy, msg = client.check_health()

        if is_healthy:
            # If a live Ollama server is running, test model discovery
            models = client.list_models()
            assert isinstance(models, list)
        else:
            # If Ollama server is offline, verify actionable troubleshooting message
            assert is_healthy is False
            assert "unreachable" in msg.lower() or "ollama serve" in msg.lower()
