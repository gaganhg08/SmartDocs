"""SmartDocs - Generative AI Document Assistant package."""

import os
os.environ.setdefault("TQDM_DISABLE", "1")
try:
    import tqdm
    tqdm.tqdm.monitor_interval = 0
except Exception:
    pass

from src.document_processor import DocumentProcessor, PDFValidationError
from src.embeddings import EmbeddingManager, SmartDocsEmbeddings
from src.vector_store import VectorStoreManager
from src.rag_pipeline import (
    RAGPipeline,
    OllamaClient,
    OllamaError,
    OllamaServerUnavailableError,
    OllamaModelNotFoundError,
    SYSTEM_PROMPT,
)

from src.summarizer import DocumentSummarizer, SummarizationError

__all__ = [
    "DocumentProcessor",
    "PDFValidationError",
    "EmbeddingManager",
    "SmartDocsEmbeddings",
    "VectorStoreManager",
    "RAGPipeline",
    "OllamaClient",
    "OllamaError",
    "OllamaServerUnavailableError",
    "OllamaModelNotFoundError",
    "SYSTEM_PROMPT",
    "DocumentSummarizer",
    "SummarizationError",
]
