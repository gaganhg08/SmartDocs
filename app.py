"""SmartDocs - Streamlit Web Application Interface.

A production-grade, local Retrieval-Augmented Generation (RAG) assistant for PDF
documents using PyMuPDF, Sentence Transformers, FAISS, and Ollama.
Includes grounded question answering and section-by-section document summarization.
"""

import os
from typing import Dict, Any, List, Optional
import streamlit as st
from dotenv import load_dotenv

# Load configuration from .env file
load_dotenv()

from src.document_processor import DocumentProcessor, PDFValidationError
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.rag_pipeline import (
    RAGPipeline,
    OllamaClient,
    OllamaServerUnavailableError,
    OllamaModelNotFoundError,
    OllamaError,
)
from src.summarizer import DocumentSummarizer

# Configuration Defaults from Environment
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))
DEFAULT_MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "20"))


@st.cache_resource(show_spinner="Loading Sentence Transformers embedding model...")
def get_embedding_manager(model_name: str, device: str = "cpu") -> EmbeddingManager:
    """Load and cache the Sentence Transformer embedding model instance.

    Guarantees the model is loaded once and reused across application reruns.
    """
    manager = EmbeddingManager(model_name=model_name, device=device)
    manager.load_model()
    return manager


def init_session_state() -> None:
    """Initialize session state variables for session isolation."""
    if "documents" not in st.session_state:
        # Map of filename -> metadata & chunks
        st.session_state.documents = {}
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStoreManager()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_filenames" not in st.session_state:
        st.session_state.processed_filenames = set()
    if "deleted_filenames" not in st.session_state:
        st.session_state.deleted_filenames = set()
    if "document_summaries" not in st.session_state:
        st.session_state.document_summaries = {}
    if "selected_doc_for_summary" not in st.session_state:
        st.session_state.selected_doc_for_summary = None


def setup_page_config() -> None:
    """Configure browser title, icon, and layout."""
    st.set_page_config(
        page_title="SmartDocs – Generative AI Document Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def rebuild_vector_index(embedding_manager: EmbeddingManager) -> None:
    """Rebuild the FAISS index using all valid chunks in the active session."""
    all_chunks: List[Dict[str, Any]] = []
    for doc_data in st.session_state.documents.values():
        all_chunks.extend(doc_data.get("chunks", []))

    if all_chunks:
        st.session_state.vector_store.build_from_chunks(all_chunks, embedding_manager)
    else:
        st.session_state.vector_store.clear()


def render_header() -> None:
    """Render application title, caption, and subtitle."""
    st.title("SmartDocs")
    st.caption("Generative AI Document Assistant")
    st.markdown("Upload documents, ask questions, and explore answers grounded in your files.")
    st.divider()


def render_sidebar(embedding_manager: EmbeddingManager) -> Dict[str, Any]:
    """Render upload, document management, and model configuration in the sidebar.

    Args:
        embedding_manager: Initialized EmbeddingManager for vector operations.

    Returns:
        Dict containing current user configuration and uploaded files.
    """
    with st.sidebar:
        st.header("📄 Document Ingestion")

        # 1. PDF Upload widget
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Select one or more PDF files. Maximum size: {DEFAULT_MAX_FILE_SIZE_MB:.0f} MB per file.",
        )

        st.divider()

        # 2. Uploaded Documents List & Controls
        st.subheader("📁 Session Documents")
        doc_count = len(st.session_state.documents)
        st.caption(f"Currently indexed: **{doc_count}** document(s)")

        if doc_count > 0:
            for fname, info in list(st.session_state.documents.items()):
                col_info, col_actions = st.columns([3, 2])
                with col_info:
                    status = info.get("status", "success")
                    if status == "success":
                        badge = "🟢 Indexed"
                    elif status == "warning":
                        badge = "⚠️ Scanned (No Text)"
                    else:
                        badge = "❌ Error"

                    st.markdown(
                        f"**{fname}**\n\n"
                        f"<small>{info.get('page_count', 0)} pgs • {len(info.get('chunks', []))} chunks • {badge}</small>",
                        unsafe_allow_html=True,
                    )
                with col_actions:
                    c_sum, c_del = st.columns(2)
                    with c_sum:
                        if st.button("📝", key=f"sum_side_{fname}", help=f"Summarize '{fname}'"):
                            st.session_state.selected_doc_for_summary = fname
                            st.rerun()
                    with c_del:
                        if st.button("🗑️", key=f"del_{fname}", help=f"Remove '{fname}' from index"):
                            del st.session_state.documents[fname]
                            st.session_state.processed_filenames.discard(fname)
                            st.session_state.deleted_filenames.add(fname)
                            st.session_state.document_summaries.pop(fname, None)
                            try:
                                rebuild_vector_index(embedding_manager)
                                st.toast(f"Removed '{fname}' and rebuilt vector index.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error rebuilding index: {e}")

            if st.button("🧹 Clear All Session Data", type="secondary", use_container_width=True):
                st.session_state.documents.clear()
                st.session_state.processed_filenames.clear()
                st.session_state.deleted_filenames.clear()
                st.session_state.document_summaries.clear()
                st.session_state.selected_doc_for_summary = None
                st.session_state.vector_store.clear()
                st.session_state.chat_history.clear()
                st.toast("Session cleared completely.")
                st.rerun()
        else:
            st.info("No documents uploaded in this session.")

        st.divider()

        # 3. Model Configuration & Connection Status
        st.subheader("🤖 Model & Inference")
        ollama_url = st.text_input(
            "Ollama Server URL",
            value=DEFAULT_OLLAMA_URL,
            help="HTTP endpoint of the local Ollama server.",
        )
        ollama_model = st.text_input(
            "Selected Model",
            value=DEFAULT_OLLAMA_MODEL,
            help="Model identifier installed in Ollama (e.g. llama3.2, mistral).",
        )

        # Connection health check
        client = OllamaClient(base_url=ollama_url)
        if st.button("🔌 Test Ollama Connection", use_container_width=True):
            is_healthy, msg = client.check_health()
            if is_healthy:
                st.success("✅ Ollama server is running and reachable!")
                try:
                    models = client.list_models()
                    if models:
                        st.caption(f"Installed models: {', '.join(models)}")
                    else:
                        st.warning("No models found in Ollama. Run `ollama pull llama3.2`.")
                except Exception:
                    pass
            else:
                st.error(f"❌ {msg}")

        # Advanced Settings Expander
        with st.expander("⚙️ Advanced Retrieval Settings", expanded=False):
            chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=250,
                max_value=2000,
                value=DEFAULT_CHUNK_SIZE,
                step=50,
                help="Maximum character length of each passage chunk.",
            )
            chunk_overlap = st.slider(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=min(500, chunk_size - 50),
                value=min(DEFAULT_CHUNK_OVERLAP, chunk_size - 50),
                step=25,
                help="Character overlap between consecutive chunks.",
            )
            top_k = st.slider(
                "Top-K Passages",
                min_value=1,
                max_value=10,
                value=DEFAULT_TOP_K,
                step=1,
                help="Number of relevant chunks retrieved for grounding.",
            )

        st.divider()
        st.caption("SmartDocs • Local RAG Architecture")

    return {
        "uploaded_files": uploaded_files or [],
        "ollama_url": ollama_url,
        "ollama_model": ollama_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
    }


def process_uploaded_files(
    uploaded_files: list,
    processor: DocumentProcessor,
    embedding_manager: EmbeddingManager,
) -> None:
    """Process newly uploaded PDF documents, chunk text, and index in FAISS.

    Displays step-by-step progress and status in the main area.

    Args:
        uploaded_files: List of uploaded file objects from the sidebar uploader.
        processor: Configured DocumentProcessor instance.
        embedding_manager: Configured EmbeddingManager instance.
    """
    if not uploaded_files:
        return

    # Synchronize deleted files tracking
    current_uploader_names = {f.name for f in uploaded_files}
    st.session_state.deleted_filenames = {
        name for name in st.session_state.deleted_filenames if name in current_uploader_names
    }

    new_files = [
        f for f in uploaded_files
        if f.name not in st.session_state.processed_filenames
        and f.name not in st.session_state.deleted_filenames
    ]

    if not new_files:
        return

    with st.status(f"Processing and indexing {len(new_files)} new document(s)...", expanded=True) as status_box:
        index_updated = False

        for uploaded_file in new_files:
            filename = uploaded_file.name
            st.write(f"📖 Parsing and extracting text from `{filename}` with PyMuPDF...")

            try:
                file_bytes = uploaded_file.getvalue()
                result = processor.process_file(file_bytes, filename)

                if result["is_scanned"]:
                    st.warning(f"⚠️ **{filename}**: {result['ocr_warning']}")
                    st.session_state.documents[filename] = {
                        "filename": filename,
                        "page_count": result["page_count"],
                        "total_chars": 0,
                        "pages": result.get("pages", []),
                        "is_scanned": True,
                        "ocr_warning": result["ocr_warning"],
                        "chunks": [],
                        "file_size_kb": len(file_bytes) / 1024,
                        "status": "warning",
                    }
                else:
                    st.write(
                        f"✂️ Split `{filename}` into {result['chunk_count']} semantic chunks "
                        f"across {result['page_count']} page(s)."
                    )
                    st.session_state.documents[filename] = {
                        "filename": filename,
                        "page_count": result["page_count"],
                        "total_chars": result["total_chars"],
                        "pages": result.get("pages", []),
                        "is_scanned": False,
                        "ocr_warning": None,
                        "chunks": result["chunks"],
                        "file_size_kb": len(file_bytes) / 1024,
                        "status": "success",
                    }
                    index_updated = True

                st.session_state.processed_filenames.add(filename)

            except PDFValidationError as e:
                st.error(f"❌ **Validation Failed for '{filename}'**: {e}")
            except Exception as e:
                st.error(f"❌ **Error processing '{filename}'**: {e}")

        if index_updated:
            st.write("🧠 Generating dense embeddings and rebuilding FAISS vector index...")
            rebuild_vector_index(embedding_manager)
            status_box.update(
                label=f"Successfully indexed {len(new_files)} document(s)!",
                state="complete",
                expanded=False,
            )
        else:
            status_box.update(
                label="Document processing completed with warnings or errors.",
                state="error",
                expanded=True,
            )


def render_chat_section(
    pipeline: RAGPipeline,
    vector_store: VectorStoreManager,
) -> None:
    """Render conversation history, input prompt, grounded answers, and source citations."""
    has_documents = not vector_store.is_empty

    # Empty State Guidance when no documents are uploaded
    if not has_documents and len(st.session_state.documents) == 0:
        st.info(
            "👈 **No documents uploaded yet.**\n\n"
            "To begin exploring your files:\n"
            "1. Upload one or more text-based PDF documents using the **Document Ingestion** area in the sidebar on the left.\n"
            "2. SmartDocs will automatically extract text page-by-page, generate dense embeddings, and build your FAISS index.\n"
            "3. Once indexed, the question input below will unlock so you can ask natural-language questions with verified citations."
        )
    elif not has_documents and len(st.session_state.documents) > 0:
        st.warning(
            "⚠️ Uploaded document(s) in this session contain no extractable text (e.g. scanned image-only PDFs). "
            "Please upload a text-based PDF to enable question answering."
        )

    # Render previous conversation history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Render retrieved sources if available
            sources = message.get("sources")
            if sources:
                with st.expander(f"📚 Retrieved Sources ({len(sources)} passages)", expanded=False):
                    for idx, s in enumerate(sources, 1):
                        st.markdown(
                            f"**Passage {idx}** — Document: `{s.get('source')}` | Page: **{s.get('page')}** "
                            f"(Relevance Score: `{s.get('relevance', 0.0):.2f}`)"
                        )
                        st.caption(f"> \"{s.get('text', '')}\"")
                        if idx < len(sources):
                            st.divider()

    # Question Input
    placeholder_text = (
        "Ask a question based on your uploaded documents..."
        if has_documents
        else "⚠️ Upload and process at least one readable PDF in the sidebar to enable questions."
    )

    query = st.chat_input(placeholder_text, disabled=not has_documents)

    if query:
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate grounded response
        with st.chat_message("assistant"):
            with st.spinner("Searching document passages and generating grounded answer..."):
                try:
                    result = pipeline.answer_query(query, vector_store)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    # Display sources or insufficient evidence guidance
                    if sources:
                        with st.expander(f"📚 Retrieved Sources ({len(sources)} passages)", expanded=True):
                            for idx, s in enumerate(sources, 1):
                                st.markdown(
                                    f"**Passage {idx}** — Document: `{s.get('source')}` | Page: **{s.get('page')}** "
                                    f"(Relevance Score: `{s.get('relevance', 0.0):.2f}`)"
                                )
                                st.caption(f"> \"{s.get('text', '')}\"")
                                if idx < len(sources):
                                    st.divider()
                    else:
                        st.info("ℹ️ No relevant passages could be retrieved for this query.")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except OllamaServerUnavailableError as e:
                    err_msg = (
                        f"❌ **Ollama Server Unavailable**\n\n"
                        f"{e}\n\n"
                        "**Troubleshooting Steps:**\n"
                        "1. Open your terminal and run: `ollama serve`\n"
                        "2. Ensure Ollama is running at the configured URL.\n"
                        "3. Click **Test Ollama Connection** in the sidebar to verify."
                    )
                    st.error(err_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": err_msg,
                        "sources": [],
                    })
                except OllamaModelNotFoundError as e:
                    err_msg = (
                        f"❌ **Ollama Model Not Found**\n\n"
                        f"{e}\n\n"
                        f"Run `ollama pull {pipeline.model_name}` in your terminal to install this model."
                    )
                    st.error(err_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": err_msg,
                        "sources": [],
                    })
                except Exception as e:
                    err_msg = f"❌ **An unexpected error occurred during generation**: {e}"
                    st.error(err_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": err_msg,
                        "sources": [],
                    })


def render_summarizer_section(summarizer: DocumentSummarizer) -> None:
    """Render document summarization interface, executive summary, and section breakdowns."""
    doc_names = list(st.session_state.documents.keys())
    if not doc_names:
        st.info(
            "📂 **No documents available for summarization.**\n\n"
            "Please upload one or more text-based PDF documents using the **Document Ingestion** area in the sidebar on the left."
        )
        return

    # Select document
    default_index = 0
    if st.session_state.selected_doc_for_summary in doc_names:
        default_index = doc_names.index(st.session_state.selected_doc_for_summary)

    selected_doc = st.selectbox(
        "Select a document to summarize:",
        options=doc_names,
        index=default_index,
        help="Choose an uploaded document to generate a structured AI summary.",
    )

    doc_info = st.session_state.documents.get(selected_doc, {})
    page_count = doc_info.get("page_count", 0)
    total_chars = doc_info.get("total_chars", 0)
    is_scanned = doc_info.get("is_scanned", False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pages", f"{page_count}")
    col2.metric("Extracted Characters", f"{total_chars:,}")
    col3.metric("Document Status", "Scanned (No Text)" if is_scanned else "Readable Text")

    if is_scanned:
        st.warning(
            f"⚠️ **Cannot summarize '{selected_doc}'**: This document contains no selectable text (scanned image-only). "
            "Optical Character Recognition (OCR) is required."
        )
        return

    # Summarize Button
    if st.button(f"✨ Summarize '{selected_doc}'", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing '{selected_doc}' and synthesizing executive summary..."):
            try:
                res = summarizer.summarize_document(doc_info)
                st.session_state.document_summaries[selected_doc] = res
                st.toast(f"Summary generated for '{selected_doc}'!")
            except OllamaServerUnavailableError as e:
                st.error(
                    f"❌ **Ollama Server Unavailable**\n\n{e}\n\n"
                    "Ensure Ollama is running (`ollama serve`) and click 'Test Ollama Connection' in the sidebar."
                )
            except OllamaModelNotFoundError as e:
                st.error(
                    f"❌ **Model Not Found in Ollama**\n\n{e}\n\n"
                    f"Run `ollama pull {summarizer.model_name}` in your terminal to download it."
                )
            except Exception as e:
                st.error(f"❌ **Summarization Failed**: {e}")

    # Display existing summary if available
    summary_data = st.session_state.document_summaries.get(selected_doc)
    if summary_data:
        st.divider()
        st.markdown(f"### 📋 Executive Summary: `{selected_doc}`")

        proc_pages = summary_data.get("processed_pages", [])
        total_p = summary_data.get("page_count", page_count)
        page_range_info = (
            f"Pages {min(proc_pages)} to {max(proc_pages)}"
            if proc_pages
            else f"Pages 1 to {total_p}"
        )

        st.caption(
            f"🤖 **Generated AI Summary** • Source Document: `{selected_doc}` • "
            f"Coverage: {page_range_info} ({len(proc_pages)} of {total_p} pages processed) • "
            f"Model: `{summary_data.get('model_used')}`"
        )

        # Show warning if partial
        if summary_data.get("is_partial") and summary_data.get("warning"):
            st.warning(summary_data["warning"])

        st.markdown(summary_data["summary"])

        # Multi-section breakdown
        section_summaries = summary_data.get("section_summaries")
        if section_summaries:
            with st.expander(f"📑 View Section-by-Section Summaries ({len(section_summaries)} sections)", expanded=False):
                for sec in section_summaries:
                    st.markdown(f"**Section {sec['section_index']} ({sec['page_range_str']})**")
                    st.markdown(sec["summary"])
                    st.divider()

        # View original extracted text expander
        with st.expander("📖 Inspect Original Extracted Text (Page by Page)", expanded=False):
            pages = doc_info.get("pages", [])
            for p in pages:
                st.markdown(f"**Page {p.get('page_number')}** ({p.get('char_count', len(p.get('text', '')))} characters):")
                st.caption(
                    f"> {p.get('text', '').strip()[:800]}..."
                    if len(p.get('text', '').strip()) > 800
                    else f"> {p.get('text', '').strip()}"
                )
                st.divider()


def main() -> None:
    """Streamlit application main entry point."""
    setup_page_config()
    init_session_state()

    # Initialize Embedding Manager (cached singleton)
    try:
        embedding_manager = get_embedding_manager(DEFAULT_EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"❌ Failed to load Sentence Transformers embedding model: {e}")
        st.stop()

    # Load Sidebar Configuration & Uploads
    config = render_sidebar(embedding_manager)

    # Initialize Document Processor with user settings
    processor = DocumentProcessor(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        max_file_size_mb=DEFAULT_MAX_FILE_SIZE_MB,
    )

    # Initialize RAG Pipeline with user settings
    pipeline = RAGPipeline(
        ollama_base_url=config["ollama_url"],
        model_name=config["ollama_model"],
        temperature=0.0,
        top_k=config["top_k"],
    )

    # Initialize Document Summarizer
    summarizer = DocumentSummarizer(
        ollama_client=pipeline.ollama_client,
        model_name=config["ollama_model"],
        max_section_chars=3500,
        temperature=0.0,
    )

    # Render Main View Header
    render_header()

    # Process newly uploaded files (progress displayed in main area)
    if config["uploaded_files"]:
        process_uploaded_files(config["uploaded_files"], processor, embedding_manager)

    # Tabs for Question Answering and Document Summarization
    tab_qa, tab_summary = st.tabs(["💬 Natural-Language Q&A", "📝 Document Summarizer"])

    with tab_qa:
        render_chat_section(pipeline, st.session_state.vector_store)

    with tab_summary:
        render_summarizer_section(summarizer)


if __name__ == "__main__":
    main()
