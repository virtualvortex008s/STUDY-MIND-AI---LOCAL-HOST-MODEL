"""
config.py — Centralized configuration for StudyMind AI.

All tunable constants live here so every other module imports
from one source of truth instead of scattering magic values.
"""

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5"          # must already be pulled via `ollama pull qwen2.5`

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"  # local folder; created on first run
CHROMA_COLLECTION  = "studymind"    # name of the vector collection

# ── Text chunking ─────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000   # characters per chunk
CHUNK_OVERLAP = 200    # overlap keeps context across chunk boundaries

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS = 5      # how many chunks to pull for each query

# ── Confidence threshold ──────────────────────────────────────────────────────
# ChromaDB returns distances (lower = more similar). Chunks whose distance
# exceeds this threshold are considered low-confidence and trigger a warning.
LOW_CONFIDENCE_THRESHOLD = 1.5

# ── PDF export ────────────────────────────────────────────────────────────────
EXPORT_FONT_SIZE_BODY  = 11
EXPORT_FONT_SIZE_H1    = 18
EXPORT_FONT_SIZE_H2    = 14
EXPORT_LEFT_MARGIN     = 20   # mm
EXPORT_RIGHT_MARGIN    = 20   # mm
EXPORT_TOP_MARGIN      = 20   # mm

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE   = "StudyMind AI"
APP_ICON    = "🧠"
APP_VERSION = "1.0.0"
