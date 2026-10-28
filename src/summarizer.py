"""Document summarization module for SmartDocs.

Provides section-by-section and hierarchical (map-reduce) summarization of multi-page
PDF documents using the existing PyMuPDF extraction pipeline and Ollama LLM.
"""

from typing import Any, Dict, List, Optional
from src.rag_pipeline import (
    OllamaClient,
    OllamaServerUnavailableError,
    OllamaModelNotFoundError,
    OllamaError,
)

SUMMARIZER_SYSTEM_PROMPT = (
    "You are SmartDocs Summarizer, an expert document analyst. "
    "Your objective is to provide a clear, objective, and well-structured summary of the supplied document text. "
    "Highlight the core objectives, key arguments, critical facts, and conclusions. "
    "Do not extrapolate beyond the provided text, do not invent unsupported claims, and maintain strict factual accuracy."
)


class SummarizationError(Exception):
    """Raised when document summarization fails."""
    pass


class DocumentSummarizer:
    """Handles single-pass and section-by-section document summarization."""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        model_name: str = "llama3.2",
        max_section_chars: int = 3500,
        temperature: float = 0.0,
    ) -> None:
        """Initialize DocumentSummarizer.

        Args:
            ollama_client: Configured OllamaClient instance.
            model_name: Name of the language model to use.
            max_section_chars: Threshold character length per section for multi-pass summarization.
            temperature: LLM sampling temperature.
        """
        self.ollama_client = ollama_client or OllamaClient()
        self.model_name = model_name
        self.max_section_chars = max_section_chars
        self.temperature = temperature

    def group_pages_into_sections(
        self,
        pages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Group consecutive pages into manageable sections that fit within model context.

        Args:
            pages: List of page dictionaries with 'page_number' and 'text'.

        Returns:
            List of section dictionaries containing 'page_range_str', 'pages', and 'text'.
        """
        sections: List[Dict[str, Any]] = []
        current_pages: List[int] = []
        current_texts: List[str] = []
        current_char_count = 0

        for page in pages:
            text = page.get("text", "").strip()
            if not text:
                continue

            page_num = page.get("page_number", 1)
            page_chars = len(text)

            # If adding this page exceeds max_section_chars and we already have content
            if current_char_count > 0 and (current_char_count + page_chars > self.max_section_chars):
                start_p = current_pages[0]
                end_p = current_pages[-1]
                range_str = f"Page {start_p}" if start_p == end_p else f"Pages {start_p}–{end_p}"
                sections.append({
                    "page_range_str": range_str,
                    "page_numbers": list(current_pages),
                    "text": "\n\n".join(current_texts),
                    "char_count": current_char_count,
                })
                current_pages = []
                current_texts = []
                current_char_count = 0

            current_pages.append(page_num)
            current_texts.append(f"--- [Page {page_num}] ---\n{text}")
            current_char_count += page_chars

        # Flush remaining section
        if current_pages:
            start_p = current_pages[0]
            end_p = current_pages[-1]
            range_str = f"Page {start_p}" if start_p == end_p else f"Pages {start_p}–{end_p}"
            sections.append({
                "page_range_str": range_str,
                "page_numbers": list(current_pages),
                "text": "\n\n".join(current_texts),
                "char_count": current_char_count,
            })

        return sections

    def summarize_single_section(
        self,
        text: str,
        filename: str,
        page_range_str: str,
    ) -> str:
        """Summarize text that fits in a single context window pass.

        Args:
            text: Document text to summarize.
            filename: Source document filename.
            page_range_str: String describing page numbers (e.g. 'Pages 1–3').

        Returns:
            Generated summary string.
        """
        prompt = (
            f"Please synthesize and summarize the following content from '{filename}' ({page_range_str}).\n\n"
            f"<DOCUMENT_TEXT>\n{text}\n</DOCUMENT_TEXT>\n\n"
            "Format the summary with the following clear sections:\n"
            "- **Executive Overview**: High-level summary of the document content.\n"
            "- **Key Points & Findings**: Detailed bulleted list of essential arguments, facts, and metrics.\n"
            "- **Core Conclusions**: Main takeaways or concluding insights."
        )

        return self.ollama_client.generate(
            prompt=prompt,
            system=SUMMARIZER_SYSTEM_PROMPT,
            model=self.model_name,
            temperature=self.temperature,
        )

    def summarize_multi_section(
        self,
        sections: List[Dict[str, Any]],
        filename: str,
        total_pages: int,
    ) -> Dict[str, Any]:
        """Summarize a long document in two stages: section summaries followed by final synthesis.

        Args:
            sections: List of grouped section dictionaries.
            filename: Document filename.
            total_pages: Total number of pages in the document.

        Returns:
            Dict containing final summary and individual section summaries.
        """
        section_summaries: List[Dict[str, Any]] = []

        # Stage 1: Summarize each manageable section
        for idx, section in enumerate(sections, 1):
            sec_prompt = (
                f"Summarize the key information and essential points from this section of '{filename}' "
                f"({section['page_range_str']}):\n\n"
                f"<SECTION_TEXT>\n{section['text']}\n</SECTION_TEXT>\n\n"
                "Provide a concise summary highlighting the main concepts, findings, and data from this section."
            )
            sec_summary = self.ollama_client.generate(
                prompt=sec_prompt,
                system=SUMMARIZER_SYSTEM_PROMPT,
                model=self.model_name,
                temperature=self.temperature,
            )
            section_summaries.append({
                "section_index": idx,
                "page_range_str": section["page_range_str"],
                "summary": sec_summary.strip(),
            })

        # Stage 2: Synthesis pass over combined section summaries
        sections_block = "\n\n".join([
            f"### Section {s['section_index']} ({s['page_range_str']}):\n{s['summary']}"
            for s in section_summaries
        ])

        synthesis_prompt = (
            f"Below are section summaries extracted from '{filename}' spanning a total of {total_pages} page(s).\n\n"
            f"<SECTION_SUMMARIES>\n{sections_block}\n</SECTION_SUMMARIES>\n\n"
            "Synthesize these section summaries into a single, cohesive final executive summary for the complete document.\n"
            "Structure your output as follows:\n"
            "- **Executive Overview**: Comprehensive high-level synopsis of the entire document.\n"
            "- **Key Points & Cross-Section Findings**: Unified bulleted list of essential findings across all sections.\n"
            "- **Major Conclusions**: Overarching implications or takeaways."
        )

        final_summary = self.ollama_client.generate(
            prompt=synthesis_prompt,
            system=SUMMARIZER_SYSTEM_PROMPT,
            model=self.model_name,
            temperature=self.temperature,
        )

        return {
            "final_summary": final_summary.strip(),
            "section_summaries": section_summaries,
        }

    def summarize_document(
        self,
        document_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize an uploaded document using its extracted pages.

        Handles empty documents, partial pages, and long documents gracefully.

        Args:
            document_data: Dictionary containing document metadata and 'pages' list.

        Returns:
            Dict containing:
                - filename: str
                - page_count: int
                - processed_pages: List[int]
                - skipped_pages: List[int]
                - is_partial: bool
                - warning: Optional[str]
                - summary: str
                - section_summaries: Optional[List[Dict[str, Any]]]
                - model_used: str

        Raises:
            ValueError: If document contains no extractable text or is unreadable.
            OllamaServerUnavailableError: If Ollama is offline.
            OllamaModelNotFoundError: If configured model is missing.
        """
        filename = document_data.get("filename", "Unknown Document")

        # 1. Validation for empty / unreadable / scanned documents
        if not document_data:
            raise ValueError("Document data is empty.")

        if document_data.get("is_scanned"):
            raise ValueError(
                f"Cannot summarize '{filename}': Document appears to be scanned or contains only images. "
                "Text extraction requires Optical Character Recognition (OCR), which is not currently supported."
            )

        pages = document_data.get("pages", [])
        if not pages:
            raise ValueError(f"Cannot summarize '{filename}': No pages found in document.")

        # 2. Identify valid vs unextractable/empty pages
        valid_pages = [p for p in pages if p.get("text", "").strip()]
        skipped_pages = [p.get("page_number", idx + 1) for idx, p in enumerate(pages) if not p.get("text", "").strip()]

        if not valid_pages:
            raise ValueError(
                f"Cannot summarize '{filename}': Document contains {len(pages)} page(s) but no extractable text."
            )

        is_partial = (len(skipped_pages) > 0)
        warning: Optional[str] = None
        if is_partial:
            skipped_str = ", ".join(str(p) for p in skipped_pages)
            warning = (
                f"Notice: Document '{filename}' was partially processed. "
                f"Page(s) [{skipped_str}] contained no extractable text and were omitted from the summary."
            )

        total_pages = document_data.get("page_count", len(pages))
        processed_page_numbers = [p.get("page_number", idx + 1) for idx, p in enumerate(valid_pages)]
        total_chars = sum(len(p.get("text", "").strip()) for p in valid_pages)

        # 3. Choose summarization strategy based on document length
        if total_chars <= self.max_section_chars:
            # Single-pass summarization
            start_p = processed_page_numbers[0]
            end_p = processed_page_numbers[-1]
            page_range_str = f"Page {start_p}" if start_p == end_p else f"Pages {start_p}–{end_p}"
            combined_text = "\n\n".join([
                f"--- [Page {p.get('page_number', 1)}] ---\n{p.get('text', '').strip()}"
                for p in valid_pages
            ])

            summary_text = self.summarize_single_section(combined_text, filename, page_range_str)

            return {
                "filename": filename,
                "page_count": total_pages,
                "processed_pages": processed_page_numbers,
                "skipped_pages": skipped_pages,
                "is_partial": is_partial,
                "warning": warning,
                "summary": summary_text.strip(),
                "section_summaries": None,
                "model_used": self.model_name,
            }
        else:
            # Multi-section hierarchical (map-reduce) summarization
            sections = self.group_pages_into_sections(valid_pages)
            res = self.summarize_multi_section(sections, filename, total_pages)

            return {
                "filename": filename,
                "page_count": total_pages,
                "processed_pages": processed_page_numbers,
                "skipped_pages": skipped_pages,
                "is_partial": is_partial,
                "warning": warning,
                "summary": res["final_summary"],
                "section_summaries": res["section_summaries"],
                "model_used": self.model_name,
            }
