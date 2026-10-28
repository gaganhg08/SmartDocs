"""Vector store management module for SmartDocs.

Handles FAISS index initialization, semantic similarity searches with source
metadata attribution, dynamic index rebuilding on document removal, session
isolation, and safe persistence.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
from langchain_community.vectorstores import FAISS
from src.embeddings import EmbeddingManager


class VectorStoreManager:
    """Manages FAISS index lifecycle, semantic searches, and document updates."""

    def __init__(self, persist_directory: Path | str = "./data/faiss_index") -> None:
        """Initialize the vector store manager.

        Args:
            persist_directory: Directory path to persist or restore serialized indices.
        """
        self.persist_directory = Path(persist_directory)
        self._faiss_index: Optional[FAISS] = None
        self._chunks: List[Dict[str, Any]] = []

    @property
    def is_empty(self) -> bool:
        """Check whether the vector store has indexed documents."""
        if self._faiss_index is None:
            return True
        try:
            return self._faiss_index.index.ntotal == 0
        except Exception:
            return len(self._chunks) == 0

    @property
    def total_chunks(self) -> int:
        """Return the number of indexed chunks."""
        if self._faiss_index is not None:
            try:
                return self._faiss_index.index.ntotal
            except Exception:
                pass
        return len(self._chunks)

    @property
    def indexed_documents(self) -> List[str]:
        """Return a sorted list of unique document filenames in the index."""
        sources = {c["metadata"].get("source", "Unknown") for c in self._chunks}
        return sorted(list(sources))

    def build_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embedding_manager: EmbeddingManager,
    ) -> Optional[FAISS]:
        """Build or rebuild a FAISS index from document chunks and embedding manager.

        Args:
            chunks: List of chunk dictionaries containing 'text' and 'metadata'.
            embedding_manager: Initialized EmbeddingManager providing embeddings.

        Returns:
            The instantiated FAISS index instance, or None if chunks is empty.
        """
        if not chunks:
            self.clear()
            return None

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = embedding_manager.get_langchain_embeddings()

        try:
            self._faiss_index = FAISS.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
            )
            self._chunks = list(chunks)
            return self._faiss_index
        except Exception as e:
            self.clear()
            raise RuntimeError(f"Failed to build FAISS vector index: {e}") from e

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search against the FAISS index.

        Args:
            query: User's natural language question or search phrase.
            k: Maximum number of relevant passages to retrieve.

        Returns:
            List of dictionaries with 'text', 'source', 'page', 'chunk_index',
            'score' (L2 distance), and 'relevance' (normalized 0-1 scale).
        """
        if self.is_empty or not query or not query.strip():
            return []

        search_k = min(max(1, k), len(self._chunks))
        results = self._faiss_index.similarity_search_with_score(
            query.strip(),
            k=search_k,
        )

        retrieved: List[Dict[str, Any]] = []
        for doc, score in results:
            # FAISS returns L2 distance (lower distance = closer match)
            # Normalized relevance score in range (0, 1]
            relevance = 1.0 / (1.0 + max(0.0, float(score)))
            retrieved.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "chunk_index": doc.metadata.get("chunk_index", 1),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "score": float(score),
                "relevance": round(relevance, 4),
            })

        return retrieved

    def remove_document(
        self,
        filename: str,
        embedding_manager: EmbeddingManager,
    ) -> None:
        """Remove all chunks associated with a specific file and rebuild index.

        Args:
            filename: Name of the file to remove.
            embedding_manager: Initialized EmbeddingManager for index rebuilding.
        """
        remaining = [
            c for c in self._chunks
            if c["metadata"].get("source") != filename
        ]
        self.build_from_chunks(remaining, embedding_manager)

    def clear(self) -> None:
        """Wipe the FAISS index and all tracked chunks."""
        self._faiss_index = None
        self._chunks = []

    def save_local(self, directory: Optional[Path | str] = None) -> None:
        """Persist the current vector index to local disk.

        Args:
            directory: Optional override directory. Defaults to self.persist_directory.

        Raises:
            RuntimeError: If the index is empty or persistence fails.
        """
        if self.is_empty:
            raise RuntimeError("Cannot save an empty FAISS vector store.")

        target = Path(directory) if directory else self.persist_directory
        target.mkdir(parents=True, exist_ok=True)
        try:
            self._faiss_index.save_local(str(target))
        except Exception as e:
            raise RuntimeError(f"Failed to save FAISS index to '{target}': {e}") from e

    def load_local(
        self,
        directory: Optional[Path | str],
        embedding_manager: EmbeddingManager,
    ) -> FAISS:
        """Load an existing FAISS index from disk.

        Args:
            directory: Directory containing saved index files.
            embedding_manager: EmbeddingManager providing embeddings.

        Returns:
            Loaded FAISS vector store instance.

        Raises:
            FileNotFoundError: If the target directory does not exist.
            RuntimeError: If index deserialization fails.
        """
        target = Path(directory) if directory else self.persist_directory
        if not target.exists():
            raise FileNotFoundError(f"FAISS index directory '{target}' does not exist.")

        embeddings = embedding_manager.get_langchain_embeddings()
        try:
            self._faiss_index = FAISS.load_local(
                str(target),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            # Reconstruct chunk cache from the loaded FAISS docstore
            if hasattr(self._faiss_index, "docstore") and hasattr(self._faiss_index.docstore, "_dict"):
                self._chunks = [
                    {"text": doc.page_content, "metadata": doc.metadata}
                    for doc in self._faiss_index.docstore._dict.values()
                ]
            return self._faiss_index
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index from '{target}': {e}") from e
