"""
rag_engine.py — Retrieval-Augmented Generation via Ollama + Qwen2.5.

Data flow for every public function:
  user input
      → vector_store.search()          — retrieve relevant chunks
      → build_prompt()                 — wrap chunks in a grounding prompt
      → ollama.chat() / .generate()    — stream or return LLM response
      → (response text, source chunks) — returned to app.py
"""

import logging
from typing import Generator

import ollama
import requests

from config import OLLAMA_HOST, DEFAULT_MODEL, TOP_K_RESULTS
import vector_store as vs

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama_health() -> tuple[bool, str]:
    """
    Ping Ollama's REST API to confirm the service is running.

    Returns (True, model_list_str) on success, (False, error_message) on failure.
    This is called by app.py before any LLM interaction.
    """
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return True, ", ".join(models) if models else "No models pulled yet"
    except requests.exceptions.ConnectionError:
        return False, (
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            "Make sure Ollama is running (`ollama serve`)."
        )
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Internal prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    We number the snippets so the model can refer to them naturally and so
    the UI can display matching citations.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}] (File: {chunk['source']}, Page: {chunk['page']})\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


def _grounding_system_prompt() -> str:
    """
    System prompt that instructs Qwen to stay strictly within the context.

    Strict grounding is critical: without it, LLMs confidently hallucinate
    answers that sound authoritative but have no basis in the student's notes.
    """
    return (
        "You are StudyMind AI, a precise academic assistant. "
        "You MUST answer ONLY using the information in the CONTEXT block below. "
        "If the context does not contain enough information to answer the question, "
        "say exactly: 'The provided documents do not contain enough information to answer this question.' "
        "Do NOT use any external knowledge. "
        "When citing information, refer to source numbers like [Source 1], [Source 2], etc. "
        "Be concise, accurate, and student-friendly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core RAG functions
# ─────────────────────────────────────────────────────────────────────────────

def answer_query(
    query: str,
    model: str = DEFAULT_MODEL,
    stream: bool = True,
) -> tuple[Generator | str, list[dict], bool]:
    """
    Answer a free-form student question grounded in indexed documents.

    Parameters
    ----------
    query  : The student's question.
    model  : Ollama model name (default from config).
    stream : If True, returns a generator for real-time streaming in the UI.

    Returns
    -------
    response   : str (stream=False) or Generator[str] (stream=True)
    sources    : list of chunk dicts used for grounding
    low_conf   : True if retrieved chunks are low-confidence matches
    """
    # 1. Retrieve the most relevant chunks from ChromaDB
    chunks = vs.search(query, n_results=TOP_K_RESULTS)

    if not chunks:
        msg = (
            "📭 No documents are indexed yet. "
            "Please upload and index a PDF first using the sidebar."
        )
        return (msg, [], False) if not stream else (_single_token_stream(msg), [], False)

    # 2. Flag if all retrieved chunks are low-confidence
    low_conf = all(c["low_conf"] for c in chunks)

    # 3. Build the grounded prompt
    context = _build_context_block(chunks)
    user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    messages = [
        {"role": "system", "content": _grounding_system_prompt()},
        {"role": "user",   "content": user_message},
    ]

    # 4. Call Ollama
    client = ollama.Client(host=OLLAMA_HOST)

    if stream:
        # Returns a generator; the UI iterates it to display tokens progressively
        def _stream_gen():
            for part in client.chat(model=model, messages=messages, stream=True):
                yield part["message"]["content"]

        return _stream_gen(), chunks, low_conf
    else:
        resp = client.chat(model=model, messages=messages, stream=False)
        return resp["message"]["content"], chunks, low_conf


def generate_summary(model: str = DEFAULT_MODEL) -> tuple[str, list[dict]]:
    """
    Produce a structured summary of ALL indexed documents.

    We retrieve a broad set of chunks (larger n_results) to cover the
    full document scope, then ask the model to synthesise them.
    """
    # Use a generic "overview" query to pull diverse chunks from the corpus
    chunks = vs.search("main topics key concepts overview introduction", n_results=15)

    if not chunks:
        return "No documents indexed.", []

    context = _build_context_block(chunks)
    prompt = (
        "Using ONLY the CONTEXT below, write a comprehensive study summary. "
        "Structure it with: "
        "1) An overview paragraph, "
        "2) Key concepts (bullet list), "
        "3) Important details (bullet list), "
        "4) A one-sentence takeaway. "
        "Do NOT use outside knowledge.\n\n"
        f"CONTEXT:\n{context}"
    )

    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.generate(model=model, prompt=prompt)
    return resp["response"], chunks


def generate_flashcards(model: str = DEFAULT_MODEL) -> tuple[str, list[dict]]:
    """
    Generate Q&A flashcard pairs from indexed material.

    Output is Markdown so the PDF exporter and the UI can render it directly.
    """
    chunks = vs.search("definition concept term explanation example", n_results=12)

    if not chunks:
        return "No documents indexed.", []

    context = _build_context_block(chunks)
    prompt = (
        "Using ONLY the CONTEXT below, generate 15 study flashcards. "
        "Format each flashcard exactly as:\n"
        "**Q:** [question]\n**A:** [answer]\n\n"
        "Focus on key terms, definitions, and concepts. "
        "Do NOT invent information not present in the context.\n\n"
        f"CONTEXT:\n{context}"
    )

    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.generate(model=model, prompt=prompt)
    return resp["response"], chunks


def generate_quiz(model: str = DEFAULT_MODEL) -> tuple[str, list[dict]]:
    """
    Create a 10-question multiple-choice quiz from indexed material.

    Each question has four labelled options (A–D) and an answer key,
    formatted in Markdown for easy rendering and PDF export.
    """
    chunks = vs.search("question fact detail example application", n_results=12)

    if not chunks:
        return "No documents indexed.", []

    context = _build_context_block(chunks)
    prompt = (
        "Using ONLY the CONTEXT below, create a 10-question multiple-choice quiz. "
        "For each question use this exact format:\n\n"
        "**Q1.** [Question text]\n"
        "A) [Option]\n"
        "B) [Option]\n"
        "C) [Option]\n"
        "D) [Option]\n"
        "✅ **Answer:** [Letter) Correct option]\n\n"
        "Vary the difficulty. Do NOT use information outside the context.\n\n"
        f"CONTEXT:\n{context}"
    )

    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.generate(model=model, prompt=prompt)
    return resp["response"], chunks


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _single_token_stream(text: str) -> Generator:
    """Yield *text* as a single chunk — lets callers treat it like a stream."""
    yield text


def list_local_models() -> list[str]:
    """Return all model names currently available in the local Ollama instance."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return [DEFAULT_MODEL]
