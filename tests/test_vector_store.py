"""Tests for FAISS vector store manager in SmartDocs."""

import os
import shutil
import pytest
from pathlib import Path
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


@pytest.fixture(scope="module")
def embedding_manager():
    manager = EmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2")
    manager.load_model()
    return manager


@pytest.fixture
def sample_chunks():
    return [
        {
            "text": "The solar system contains eight planets orbiting the sun.",
            "metadata": {"source": "astronomy.pdf", "page": 1, "chunk_index": 1, "chunk_id": "astronomy_p1_c1"},
        },
        {
            "text": "Mars is known as the red planet due to iron oxide on its surface.",
            "metadata": {"source": "astronomy.pdf", "page": 2, "chunk_index": 2, "chunk_id": "astronomy_p2_c2"},
        },
        {
            "text": "Photosynthesis is the chemical process plants use to convert sunlight into energy.",
            "metadata": {"source": "biology.pdf", "page": 1, "chunk_index": 1, "chunk_id": "biology_p1_c1"},
        },
    ]


class TestVectorStoreManager:
    """Tests covering FAISS vector indexing, retrieval, and document management."""

    def test_build_and_search(self, embedding_manager, sample_chunks):
        vs = VectorStoreManager()
        assert vs.is_empty is True
        assert vs.total_chunks == 0

        vs.build_from_chunks(sample_chunks, embedding_manager)
        assert vs.is_empty is False
        assert vs.total_chunks == 3
        assert vs.indexed_documents == ["astronomy.pdf", "biology.pdf"]

        # Search for Mars
        results = vs.similarity_search("Tell me about the red planet Mars", k=1)
        assert len(results) == 1
        best_match = results[0]
        assert "Mars" in best_match["text"]
        assert best_match["source"] == "astronomy.pdf"
        assert best_match["page"] == 2
        assert 0.0 < best_match["relevance"] <= 1.0

    def test_empty_search_handling(self, embedding_manager, sample_chunks):
        vs = VectorStoreManager()
        # Search against empty index
        assert vs.similarity_search("planets", k=2) == []

        # Build index, search with empty query string
        vs.build_from_chunks(sample_chunks, embedding_manager)
        assert vs.similarity_search("", k=2) == []
        assert vs.similarity_search("   ", k=2) == []

    def test_remove_document_and_rebuild(self, embedding_manager, sample_chunks):
        vs = VectorStoreManager()
        vs.build_from_chunks(sample_chunks, embedding_manager)
        assert "astronomy.pdf" in vs.indexed_documents
        assert "biology.pdf" in vs.indexed_documents
        assert vs.total_chunks == 3

        # Remove astronomy.pdf
        vs.remove_document("astronomy.pdf", embedding_manager)
        assert vs.total_chunks == 1
        assert vs.indexed_documents == ["biology.pdf"]

        # Searching for astronomy now yields no astronomy results
        results = vs.similarity_search("orbiting the sun", k=2)
        for r in results:
            assert r["source"] != "astronomy.pdf"

    def test_clear_vector_store(self, embedding_manager, sample_chunks):
        vs = VectorStoreManager()
        vs.build_from_chunks(sample_chunks, embedding_manager)
        assert vs.is_empty is False

        vs.clear()
        assert vs.is_empty is True
        assert vs.total_chunks == 0
        assert vs.indexed_documents == []

    def test_save_and_load_persistence(self, embedding_manager, sample_chunks, tmp_path):
        vs = VectorStoreManager(persist_directory=tmp_path / "faiss_test_idx")
        vs.build_from_chunks(sample_chunks, embedding_manager)

        # Save index
        vs.save_local()
        assert (tmp_path / "faiss_test_idx" / "index.faiss").exists()

        # Load into new manager
        vs_loaded = VectorStoreManager()
        vs_loaded.load_local(tmp_path / "faiss_test_idx", embedding_manager)
        assert vs_loaded.is_empty is False

        # Verify search works on loaded index
        loaded_res = vs_loaded.similarity_search("photosynthesis sunlight", k=1)
        assert len(loaded_res) == 1
        assert loaded_res[0]["source"] == "biology.pdf"
