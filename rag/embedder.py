"""
PART B: Embedding Pipeline
Uses sentence-transformers (all-MiniLM-L6-v2) — lightweight, fast, strong
on semantic similarity tasks. No external API key required.

Student: Maame Yaa Adumaba Appiah | Index: 10022200171
Course : CS4241 - Introduction to Artificial Intelligence | ACity 2026
"""
from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Return a 2-D float32 array of shape (n, dim)."""
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity via dot product
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string, shape (1, dim)."""
    return embed_texts([query])
