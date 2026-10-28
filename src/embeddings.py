"""Embedding management module for SmartDocs.

Manages loading Sentence Transformers models locally, vectorizing text passages,
and providing LangChain-compatible embedding wrappers.
"""

import os
# Prevent tqdm monitor thread crash on Windows Python 3.13
os.environ.setdefault("TQDM_DISABLE", "1")

from typing import Any, List, Optional
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class SmartDocsEmbeddings(Embeddings):
    """LangChain Embeddings adapter wrapping an existing EmbeddingManager instance.

    Avoids loading duplicate model instances in memory.
    """

    def __init__(self, embedding_manager: "EmbeddingManager") -> None:
        self.embedding_manager = embedding_manager

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Vectorize a list of document chunk texts."""
        return self.embedding_manager.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        """Vectorize a user search query."""
        return self.embedding_manager.embed_query(text)


class EmbeddingManager:
    """Manages local Sentence Transformer embedding model lifecycle and inference."""

    _cached_instance: Optional["EmbeddingManager"] = None

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        """Initialize the embedding manager.

        Args:
            model_name: HuggingFace model identifier or local directory.
            device: Hardware execution target ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = device
        self._model: Optional[SentenceTransformer] = None
        self._langchain_adapter: Optional[SmartDocsEmbeddings] = None

    def load_model(self) -> SentenceTransformer:
        """Load or retrieve the Sentence Transformer embedding model.

        Tries local cached files first to minimize network overhead and sandbox delays,
        falling back to online download if necessary.

        Returns:
            Instantiated SentenceTransformer model.

        Raises:
            RuntimeError: If model cannot be loaded or downloaded.
        """
        if self._model is not None:
            return self._model

        # Attempt 1: Load from local cache (fast & works offline)
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=True,
            )
            return self._model
        except Exception:
            pass

        # Attempt 2: Download from HuggingFace Hub
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            return self._model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load or download embedding model '{self.model_name}': {e}. "
                "Ensure an internet connection is available or that the model identifier is valid."
            ) from e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vector embeddings for a collection of texts.

        Args:
            texts: List of text passages.

        Returns:
            List of vector embeddings as lists of floats.
        """
        if not texts:
            return []

        model = self.load_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Generate a dense vector embedding for a query string.

        Args:
            query: The user query string.

        Returns:
            A single vector embedding as a list of floats.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        model = self.load_model()
        vec = model.encode(query.strip(), convert_to_numpy=True, show_progress_bar=False)
        return vec.tolist()

    def get_langchain_embeddings(self) -> SmartDocsEmbeddings:
        """Obtain a LangChain-compatible Embeddings adapter reusing this manager.

        Returns:
            SmartDocsEmbeddings instance.
        """
        if self._langchain_adapter is None:
            self._langchain_adapter = SmartDocsEmbeddings(self)
        return self._langchain_adapter
