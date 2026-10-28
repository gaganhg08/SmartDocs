"""Tests for embedding management module in SmartDocs."""

import os
import pytest
from src.embeddings import EmbeddingManager, SmartDocsEmbeddings

# Ensure offline loading during test execution to prevent network timeouts
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


@pytest.fixture(scope="module")
def embedding_manager():
    """Shared fixture to load the embedding model once."""
    manager = EmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2")
    manager.load_model()
    return manager


class TestEmbeddingManager:
    """Tests covering Sentence Transformers embedding generation."""

    def test_embedding_dimensions(self, embedding_manager):
        query_vec = embedding_manager.embed_query("What is semantic search?")
        assert isinstance(query_vec, list)
        assert len(query_vec) == 384
        assert all(isinstance(val, float) for val in query_vec)

    def test_embed_multiple_texts(self, embedding_manager):
        texts = [
            "SmartDocs processes multi-page PDF documents.",
            "FAISS provides high-performance vector similarity search.",
        ]
        vectors = embedding_manager.embed_texts(texts)
        assert len(vectors) == 2
        assert len(vectors[0]) == 384
        assert len(vectors[1]) == 384

    def test_embed_empty_list(self, embedding_manager):
        vectors = embedding_manager.embed_texts([])
        assert vectors == []

    def test_empty_query_raises_error(self, embedding_manager):
        with pytest.raises(ValueError, match="Query string cannot be empty"):
            embedding_manager.embed_query("   ")

    def test_langchain_adapter(self, embedding_manager):
        adapter = embedding_manager.get_langchain_embeddings()
        assert isinstance(adapter, SmartDocsEmbeddings)

        docs_vecs = adapter.embed_documents(["LangChain integration test"])
        assert len(docs_vecs) == 1
        assert len(docs_vecs[0]) == 384

        q_vec = adapter.embed_query("Test query")
        assert len(q_vec) == 384
