"""PDF extraction and document ingestion for RAG pipeline."""

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF.

    Preserves page boundaries with [Page N] markers.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"[Page {page_num}]\n{text}")
    doc.close()
    return "\n\n".join(pages)


def load_documents_from_directory(directory: str) -> list[dict]:
    """Load all supported documents from a directory (recursive).

    Returns:
        List of dicts with 'text', 'source', and 'filename' keys.
    """
    documents = []
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning("Directory not found: %s", directory)
        return documents

    for file_path in sorted(dir_path.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(str(file_path))
            else:
                text = file_path.read_text(encoding="utf-8")

            if text.strip():
                documents.append({
                    "text": text,
                    "source": str(file_path),
                    "filename": file_path.name,
                })
        except Exception:
            logger.exception("Failed to load %s", file_path)

    return documents


async def ingest_directory(rag_engine, directory: str) -> int:
    """Ingest all documents from a directory into the RAG engine.

    Args:
        rag_engine: PJIRagEngine instance (already initialized).
        directory: Path to directory containing documents.

    Returns:
        Number of documents successfully ingested.
    """
    documents = load_documents_from_directory(directory)
    count = 0
    for doc in documents:
        try:
            await rag_engine.ingest(doc["text"], source=doc["source"])
            logger.info("Ingested: %s", doc["filename"])
            count += 1
        except Exception:
            logger.exception("Error ingesting %s", doc["filename"])
    return count
