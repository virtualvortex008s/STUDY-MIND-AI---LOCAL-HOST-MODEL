"""
pdf_processor.py — PDF text extraction and chunking.

Data flow:
  PDF file path(s)
      → extract_text_from_pdf()   — raw page-level text + metadata
      → clean_text()              — strips artefacts, normalises whitespace
      → chunk_document()          — splits into overlapping fixed-size chunks
      → list[dict]                — ready to hand to vector_store.py
"""

import re
import logging
from pathlib import Path
from typing import Optional

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf(file_path: str) -> tuple[list[dict], str]:
    """
    Full pipeline: extract → clean → chunk for one PDF file.

    Returns
    -------
    chunks : list[dict]
        Each dict has keys  ``text``, ``source``, ``page``, ``chunk_id``.
    warning : str
        Empty string on success; human-readable message if issues were found
        (e.g., scanned-only pages with no extractable text).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # Step 1 – pull raw text from every page
    pages, warning = extract_text_from_pdf(str(path))

    if not pages:
        return [], warning or "No extractable text found. The PDF may be scanned/image-only."

    # Step 2 – join pages into one document; we keep page-boundary markers
    # so we can recover page numbers later
    full_text, page_map = _assemble_with_markers(pages)

    # Step 3 – clean the assembled text
    clean = clean_text(full_text)

    # Step 4 – split into overlapping chunks
    chunks = chunk_document(clean, source=path.name, page_map=page_map)

    return chunks, warning


def extract_text_from_pdf(file_path: str) -> tuple[list[dict], str]:
    """
    Extract per-page text from a PDF using pdfplumber.

    pdfplumber is preferred over pypdf because it handles complex layouts,
    tables, and multi-column text far better.

    Returns a list of page dicts:
        {"page_num": int, "text": str}
    and an optional warning string.
    """
    pages: list[dict] = []
    warning = ""
    empty_pages = 0

    try:
        with pdfplumber.open(file_path) as pdf:
            total = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                try:
                    raw = page.extract_text() or ""
                except Exception as e:
                    logger.warning("Page %d extraction error: %s", i + 1, e)
                    raw = ""

                if raw.strip():
                    pages.append({"page_num": i + 1, "text": raw})
                else:
                    empty_pages += 1
                    logger.debug("Page %d yielded no text (possibly an image).", i + 1)

        if empty_pages == total:
            warning = (
                "⚠️ This PDF appears to be fully image-based or scanned. "
                "No text could be extracted. Use an OCR tool (e.g., Adobe Acrobat, "
                "tesseract) to convert it to a text-based PDF first."
            )
        elif empty_pages > 0:
            warning = (
                f"⚠️ {empty_pages}/{total} pages contained no extractable text "
                "(likely images or scanned content). Those pages were skipped."
            )

    except Exception as e:
        logger.error("Failed to open PDF %s: %s", file_path, e)
        raise RuntimeError(f"Could not open PDF: {e}") from e

    return pages, warning


def clean_text(text: str) -> str:
    """
    Normalise extracted PDF text.

    Common PDF artefacts we remove:
    • Repeated whitespace / line-break clutter from multi-column layouts
    • Soft-hyphen word breaks (e.g., "exam-\nple" → "example")
    • Form-feed characters left by some PDF renderers
    • Multiple blank lines collapsed to one
    """
    # Rejoin words split across lines by a soft hyphen
    text = re.sub(r"-\n", "", text)

    # Remove form feeds
    text = text.replace("\f", "\n")

    # Collapse runs of spaces/tabs to a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse more than two consecutive newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace on each line
    text = "\n".join(line.strip() for line in text.splitlines())

    return text.strip()


def chunk_document(
    text: str,
    source: str,
    page_map: Optional[dict] = None,
) -> list[dict]:
    """
    Split *text* into overlapping chunks suitable for embedding.

    We use LangChain's RecursiveCharacterTextSplitter which tries to split
    on paragraph → sentence → word boundaries in that order, keeping chunks
    semantically coherent.

    Each chunk dict:
        {
            "text":     str,   # chunk content
            "source":  str,   # filename
            "page":    int,   # best-guess page number (from page_map)
            "chunk_id": str,  # "<source>_chunk_<n>"
        }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Preferred split boundaries, tried in order
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)
    chunks = []

    for idx, chunk_text in enumerate(raw_chunks):
        # Estimate which page this chunk originated from by finding
        # the first page-marker whose offset precedes the chunk
        page_num = _estimate_page(chunk_text, page_map)

        chunks.append({
            "text":     chunk_text,
            "source":  source,
            "page":    page_num,
            "chunk_id": f"{source}_chunk_{idx}",
        })

    logger.info("'%s' → %d chunks", source, len(chunks))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_with_markers(pages: list[dict]) -> tuple[str, dict]:
    """
    Join page texts with sentinel strings so we can map chunks back to pages.

    Returns the full string and a dict mapping sentinel → page_num.
    """
    parts = []
    page_map = {}  # sentinel_text → page_num

    for page in pages:
        sentinel = f"\n<<<PAGE_{page['page_num']}>>>\n"
        page_map[sentinel.strip()] = page["page_num"]
        parts.append(sentinel + page["text"])

    return "\n".join(parts), page_map


def _estimate_page(chunk_text: str, page_map: Optional[dict]) -> int:
    """
    Return the page number that most likely contains this chunk.

    We search the chunk for the highest-numbered PAGE sentinel embedded
    during assembly. Falls back to 1 if no marker is found.
    """
    if not page_map:
        return 1

    best_page = 1
    for sentinel, page_num in page_map.items():
        if sentinel in chunk_text:
            best_page = max(best_page, page_num)

    return best_page
