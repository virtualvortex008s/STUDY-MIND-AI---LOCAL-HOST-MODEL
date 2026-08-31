"""
vector_store.py — Persistent ChromaDB vector store management.

Data flow:
  Chunks (list[dict] from pdf_processor)
      → add_chunks()        — embeds & stores in ChromaDB
  Query string
      → search()            — embeds query & returns nearest chunks
"""

import logging
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    TOP_K_RESULTS,
    LOW_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton client — reused across Streamlit reruns via caching
# ─────────────────────────────────────────────────────────────────────────────

_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_collection():
    """
    Return (and lazily create) the persistent ChromaDB collection.

    ChromaDB's DefaultEmbeddingFunction uses the all-MiniLM-L6-v2 model
    from sentence-transformers, which runs entirely locally — no API calls.
    The model is downloaded automatically on first use (~90 MB).
    """
    global _client, _collection

    if _collection is not None:
        return _collection

    # PersistentClient saves the vector index to disk so data survives
    # between Streamlit restarts.
    _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    _collection = _client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        # DefaultEmbeddingFunction wraps sentence-transformers locally
        embedding_function=DefaultEmbeddingFunction(),
        # cosine similarity is better than Euclidean for semantic search
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(
        "ChromaDB collection '%s' ready (%d documents).",
        CHROMA_COLLECTION,
        _collection.count(),
    )
    return _collection


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def add_chunks(chunks: list[dict]) -> int:
    """
    Embed and persist a list of text chunks.

    Parameters
    ----------
    chunks : list[dict]
        Each dict must have keys: ``text``, ``source``, ``page``, ``chunk_id``.

    Returns
    -------
    int
        Number of new chunks successfully added (duplicates are skipped).
    """
    col = _get_collection()

    # Separate fields for ChromaDB's expected format
    ids       = [c["chunk_id"] for c in chunks]
    documents = [c["text"]     for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

    # Check which IDs already exist so we don't embed twice
    existing = set(col.get(ids=ids)["ids"])
    new_mask = [i for i, cid in enumerate(ids) if cid not in existing]

    if not new_mask:
        logger.info("All %d chunks already indexed — skipping.", len(chunks))
        return 0

    col.add(
        ids       = [ids[i]       for i in new_mask],
        documents = [documents[i] for i in new_mask],
        metadatas = [metadatas[i] for i in new_mask],
    )

    added = len(new_mask)
    logger.info("Added %d new chunks to ChromaDB.", added)
    return added


def search(query: str, n_results: int = TOP_K_RESULTS) -> list[dict]:
    """
    Find the *n_results* most semantically similar chunks to *query*.

    Returns a list of result dicts:
        {
            "text":       str,
            "source":    str,
            "page":      int,
            "distance":  float,  # 0 = identical, 1 = orthogonal (cosine)
            "low_conf":  bool,   # True if distance > threshold
        }
    """
    col = _get_collection()

    if col.count() == 0:
        logger.warning("Vector store is empty. No results returned.")
        return []

    results = col.query(
        query_texts=[query],
        n_results=min(n_results, col.count()),  # can't ask for more than exist
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns lists-of-lists because it supports batch queries.
    # We only sent one query, so take index [0].
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    formatted = []
    for doc, meta, dist in zip(docs, metas, distances):
        formatted.append({
            "text":      doc,
            "source":   meta.get("source", "unknown"),
            "page":     meta.get("page", 0),
            "distance": round(dist, 4),
            "low_conf": dist > LOW_CONFIDENCE_THRESHOLD,
        })

    return formatted


def get_chunk_count() -> int:
    """Return the total number of embedded chunks across all documents."""
    return _get_collection().count()


def get_indexed_sources() -> list[str]:
    """Return unique PDF filenames that have been indexed."""
    col = _get_collection()
    if col.count() == 0:
        return []

    # Fetch all metadata (no documents needed)
    all_meta = col.get(include=["metadatas"])["metadatas"]
    sources  = sorted({m["source"] for m in all_meta if "source" in m})
    return sources


def delete_source(source_name: str) -> int:
    """
    Remove all chunks belonging to *source_name* from the vector store.

    Returns the number of chunks deleted.
    """
    col = _get_collection()

    # Query for IDs that match this source
    results = col.get(where={"source": source_name}, include=[])
    ids_to_delete = results["ids"]

    if ids_to_delete:
        col.delete(ids=ids_to_delete)
        logger.info("Deleted %d chunks for source '%s'.", len(ids_to_delete), source_name)

    return len(ids_to_delete)


def reset_collection() -> None:
    """
    Drop and recreate the ChromaDB collection — wipes all indexed data.
    Use with caution (exposed in the UI only via a confirmation step).
    """
    global _collection

    if _client is None:
        _get_collection()  # ensure client exists

    _client.delete_collection(CHROMA_COLLECTION)
    _collection = None  # force recreation on next call
    _get_collection()
    logger.warning("ChromaDB collection reset. All data wiped.")
