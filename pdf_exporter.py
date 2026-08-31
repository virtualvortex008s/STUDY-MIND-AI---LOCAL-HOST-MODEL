"""
pdf_exporter.py — Convert Markdown study content into a styled PDF.

Data flow:
  Markdown string (from rag_engine)
      → parse_markdown_to_elements()   — classify lines (h1, h2, bullet, body)
      → build_pdf()                    — render with fpdf2
      → bytes                          — returned to app.py for st.download_button
"""

import io
import re
import logging
from datetime import datetime

from fpdf import FPDF

from config import (
    EXPORT_FONT_SIZE_BODY,
    EXPORT_FONT_SIZE_H1,
    EXPORT_FONT_SIZE_H2,
    EXPORT_LEFT_MARGIN,
    EXPORT_RIGHT_MARGIN,
    EXPORT_TOP_MARGIN,
    APP_TITLE,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FPDF subclass with a styled header/footer
# ─────────────────────────────────────────────────────────────────────────────

class StudyMindPDF(FPDF):
    """Custom FPDF with branded header and page-number footer."""

    def __init__(self, doc_title: str = "Study Notes"):
        super().__init__()
        self.doc_title = doc_title
        self.set_margins(EXPORT_LEFT_MARGIN, EXPORT_TOP_MARGIN, EXPORT_RIGHT_MARGIN)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        # Thin coloured bar at the top of every page
        self.set_fill_color(30, 30, 46)          # dark navy
        self.rect(0, 0, self.w, 12, style="F")

        self.set_y(2)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(180, 180, 230)        # soft lavender text
        self.cell(0, 8, f"🧠 {APP_TITLE}  —  {self.doc_title}", align="L")

        # Date on the right
        self.set_y(2)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(140, 140, 160)
        date_str = datetime.now().strftime("%d %b %Y")
        self.cell(0, 8, date_str, align="R")

        self.set_text_color(0, 0, 0)              # reset to black for body
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown parser
# ─────────────────────────────────────────────────────────────────────────────

def _strip_markdown_inline(text: str) -> str:
    """
    Remove common inline Markdown formatting so raw text is rendered cleanly.

    fpdf2 does not render Markdown natively, so we strip markers and apply
    bold/italic via FPDF font-style calls where needed.
    """
    # Bold: **text** or __text__ → just text (we handle bold separately per-line)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    # Italic: *text* or _text_
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    # Inline code: `text`
    text = re.sub(r"`(.*?)`", r"\1", text)
    # Answer checkmarks
    text = text.replace("✅", "[Answer]")
    return text.strip()


def _classify_line(line: str) -> tuple[str, str]:
    """
    Classify a Markdown line into a type tag and its cleaned text.

    Returns (type, text) where type is one of:
        'h1', 'h2', 'h3', 'bullet', 'numbered', 'answer', 'blank', 'body'
    """
    stripped = line.strip()

    if not stripped:
        return "blank", ""
    if stripped.startswith("# "):
        return "h1", _strip_markdown_inline(stripped[2:])
    if stripped.startswith("## "):
        return "h2", _strip_markdown_inline(stripped[3:])
    if stripped.startswith("### "):
        return "h3", _strip_markdown_inline(stripped[4:])
    if re.match(r"^[-*+] ", stripped):
        return "bullet", _strip_markdown_inline(stripped[2:])
    if re.match(r"^\d+[.)]\s", stripped):
        return "numbered", _strip_markdown_inline(re.sub(r"^\d+[.)]\s", "", stripped))
    if stripped.startswith("[Answer]") or "Answer:" in stripped:
        return "answer", _strip_markdown_inline(stripped)
    # Lines that start bold (Q1., Q2. pattern from quiz output)
    if re.match(r"\*\*Q\d", stripped):
        return "question", _strip_markdown_inline(stripped)

    return "body", _strip_markdown_inline(stripped)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def markdown_to_pdf_bytes(
    markdown_text: str,
    doc_title: str = "Study Notes",
) -> bytes:
    """
    Convert *markdown_text* to a styled PDF and return it as raw bytes.

    The bytes are passed directly to Streamlit's ``st.download_button``
    without writing to disk — keeping everything in memory.
    """
    pdf = StudyMindPDF(doc_title=doc_title)
    pdf.add_page()

    # Use built-in core fonts (no external font files needed)
    # Helvetica works well for structured study notes
    lines = markdown_text.splitlines()

    for line in lines:
        kind, text = _classify_line(line)
        _render_element(pdf, kind, text)

    # Return as bytes object; FPDF.output() with dest='S' returns a str in
    # older versions, bytes in fpdf2. We normalise to bytes.
    raw = pdf.output()
    if isinstance(raw, str):
        return raw.encode("latin-1")
    return bytes(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_element(pdf: StudyMindPDF, kind: str, text: str) -> None:
    """Render a single classified element onto the PDF canvas."""

    effective_width = pdf.w - pdf.l_margin - pdf.r_margin

    if kind == "blank":
        pdf.ln(3)

    elif kind == "h1":
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", EXPORT_FONT_SIZE_H1)
        pdf.set_text_color(30, 30, 46)            # dark navy heading
        pdf.set_fill_color(230, 230, 255)         # light lavender background
        pdf.multi_cell(effective_width, 10, text, fill=True, align="L")
        pdf.ln(2)
        pdf.set_text_color(0, 0, 0)

    elif kind == "h2":
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", EXPORT_FONT_SIZE_H2)
        pdf.set_text_color(50, 50, 120)
        pdf.multi_cell(effective_width, 8, text, align="L")
        # Underline via a thin rule
        pdf.set_draw_color(100, 100, 200)
        pdf.set_line_width(0.4)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + effective_width, pdf.get_y())
        pdf.ln(2)
        pdf.set_text_color(0, 0, 0)
        pdf.set_draw_color(0, 0, 0)

    elif kind == "h3":
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", EXPORT_FONT_SIZE_BODY + 1)
        pdf.set_text_color(60, 60, 140)
        pdf.multi_cell(effective_width, 7, text, align="L")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)

    elif kind == "bullet":
        pdf.set_font("Helvetica", "", EXPORT_FONT_SIZE_BODY)
        pdf.set_x(pdf.l_margin + 5)              # indent
        # Bullet character
        pdf.cell(5, 6, "\u2022")
        pdf.multi_cell(effective_width - 10, 6, text)

    elif kind == "numbered":
        pdf.set_font("Helvetica", "", EXPORT_FONT_SIZE_BODY)
        pdf.set_x(pdf.l_margin + 5)
        pdf.multi_cell(effective_width - 5, 6, text)

    elif kind == "question":
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", EXPORT_FONT_SIZE_BODY)
        pdf.set_text_color(20, 20, 80)
        pdf.multi_cell(effective_width, 6, text)
        pdf.set_text_color(0, 0, 0)

    elif kind == "answer":
        pdf.set_font("Helvetica", "B", EXPORT_FONT_SIZE_BODY)
        pdf.set_text_color(0, 120, 60)            # green for answers
        pdf.multi_cell(effective_width, 6, text)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    else:  # body
        pdf.set_font("Helvetica", "", EXPORT_FONT_SIZE_BODY)
        pdf.multi_cell(effective_width, 6, text)
