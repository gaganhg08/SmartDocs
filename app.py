"""SmartDocs - Streamlit Web Application Interface.

This module provides the user interface for SmartDocs:
- File upload widget for PDF documents.
- Configuration sidebar for adjusting chunk sizes and retrieval settings.
- Conversational chat UI to submit queries and display grounded answers.
- Collapsible inspection drawer displaying retrieved document excerpts and citations.

Note:
    Phase 1 establishes the UI scaffolding and layout. Core RAG execution logic
    will be wired into the interface in Phase 2.
"""

import streamlit as st


def setup_page_config() -> None:
    """Configure Streamlit browser tab settings, title, and page layout."""
    st.set_page_config(
        page_title="SmartDocs – Generative AI Document Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar() -> dict:
    """Render application controls, model selectors, and settings in the sidebar.

    Returns:
        dict: User-selected configuration settings.
    """
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown(
            "Customize the Retrieval-Augmented Generation (RAG) pipeline settings."
        )

        st.subheader("Model Selection")
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["OpenAI", "Google Gemini", "Anthropic", "Ollama (Local)"],
            index=0,
            help="Choose the model provider for answering questions.",
        )

        model_name = st.text_input(
            "Model Name",
            value="gpt-4o-mini",
            help="Specify model identifier for the chosen provider.",
        )

        st.subheader("Retrieval Settings")
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=250,
            max_value=2000,
            value=1000,
            step=50,
            help="Length of document text segments for embedding.",
        )

        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=500,
            value=200,
            step=25,
            help="Number of overlapping characters between chunks.",
        )

        top_k = st.slider(
            "Top-K Passages",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Number of retrieved passages passed into the context window.",
        )

        st.divider()
        st.caption(
            "SmartDocs v0.1.0-dev • Built with Streamlit, LangChain, & FAISS"
        )

    return {
        "llm_provider": llm_provider,
        "model_name": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
    }


def render_main_content() -> None:
    """Render the primary view including project header and upload panel."""
    st.title("📄 SmartDocs – Generative AI Document Assistant")
    st.markdown(
        "Empower your PDF documents with intelligent semantic retrieval "
        "and factual question answering powered by LangChain and FAISS."
    )

    # Active development status notification
    st.info(
        "🛠️ **Project Initialization Phase**: The repository and architecture scaffold "
        "have been set up. Document ingestion, vector indexing, and RAG execution "
        "pipelines are under active development in Phase 2.",
        icon="ℹ️",
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select the documents you wish to query.",
        )

        if uploaded_files:
            st.success(f"{len(uploaded_files)} document(s) uploaded successfully.")
            for file in uploaded_files:
                st.write(f"- 📄 `{file.name}` ({file.size / 1024:.1f} KB)")

    with col2:
        st.subheader("2. Ask Questions")
        query = st.chat_input(
            "Ask a question about the uploaded documents...",
            disabled=True,  # Disabled during Phase 1 scaffold
        )
        st.caption(
            "Chat interaction will be enabled upon Phase 2 pipeline integration."
        )


def main() -> None:
    """Streamlit application entry point."""
    setup_page_config()
    _ = render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
