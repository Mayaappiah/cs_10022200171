"""
PART B: Custom Retrieval System
- FAISS IndexFlatIP (inner-product on normalized vectors = cosine similarity)
- Top-k retrieval with similarity scores
- Query expansion via synonym/keyword augmentation (extends retrieval coverage)

Student: Maame Yaa Adumaba Appiah | Index: 10022200171
Course : CS4241 - Introduction to Artificial Intelligence | ACity 2026
- Hybrid search: keyword pre-filter + vector re-rank
"""
from __future__ import annotations
import re
import numpy as np
import faiss
from rag.embedder import embed_texts, embed_query


class VectorStore:
    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[dict] = []
        self.dim: int = 0

    def build(self, chunks: list[dict]) -> None:
        """Embed all chunks and build FAISS index."""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Basic top-k cosine retrieval."""
        q_emb = embed_query(query)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.chunks[idx])
            result["score"] = float(score)
            results.append(result)
        return results

    def hybrid_retrieve(self, query: str, top_k: int = 5, keyword_pool: int = 200) -> list[dict]:
        """
        Hybrid search: keyword pre-filter to candidate pool, then vector re-rank.
        Fixes failure case where vector search misses exact terminology (e.g. year specificity).
        Strategy: prefer chunks that match ALL high-value query terms (AND-logic priority),
        fall back to ANY-match for remaining candidates.
        """
        keywords = _extract_keywords(query)
        # Separate numeric tokens (years, figures) — these need AND-logic
        numeric_kws = [kw for kw in keywords if kw.isdigit()]
        text_kws = [kw for kw in keywords if not kw.isdigit()]

        def match_score(text: str) -> int:
            t = text.lower()
            hits = sum(1 for kw in keywords if kw.lower() in t)
            # bonus for every numeric match (year precision)
            hits += sum(2 for kw in numeric_kws if kw in t)
            return hits

        if keywords:
            scored = [(i, match_score(c["text"])) for i, c in enumerate(self.chunks)]
            scored = [(i, s) for i, s in scored if s > 0]
            scored.sort(key=lambda x: -x[1])
            candidate_indices = [i for i, _ in scored]
        else:
            candidate_indices = list(range(len(self.chunks)))

        # Fall back to full index if too few candidates
        if len(candidate_indices) < top_k:
            candidate_indices = list(range(len(self.chunks)))

        # Vector re-rank over candidates
        candidate_texts = [self.chunks[i]["text"] for i in candidate_indices[:keyword_pool]]
        candidate_embs = embed_texts(candidate_texts)
        q_emb = embed_query(query)
        vector_scores = (q_emb @ candidate_embs.T).flatten()

        # Keyword boost: add a small bonus for each keyword hit (fixes year specificity)
        kw_boost = np.array([
            sum(0.05 for kw in keywords if kw.lower() in candidate_texts[j].lower())
            for j in range(len(candidate_texts))
        ], dtype=np.float32)
        scores = vector_scores + kw_boost
        sorted_pos = np.argsort(scores)[::-1][:top_k]

        results = []
        for pos in sorted_pos:
            chunk = dict(self.chunks[candidate_indices[pos]])
            chunk["score"] = float(scores[pos])
            results.append(chunk)
        return results

    def retrieve_with_expansion(self, query: str, top_k: int = 5) -> list[dict]:
        """Query expansion: augment the query with related terms before retrieval."""
        expanded = _expand_query(query)
        return self.retrieve(expanded, top_k=top_k)


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_keywords(query: str) -> list[str]:
    stopwords = {"the", "a", "an", "is", "in", "of", "and", "or", "to", "for",
                 "what", "who", "how", "when", "where", "did", "do", "does",
                 "was", "were", "with", "from", "by"}
    tokens = re.findall(r"\b\w+\b", query.lower())
    return [t for t in tokens if t not in stopwords and len(t) > 2]


_EXPANSION_MAP = {
    "election": ["vote", "presidential", "result", "candidate", "party"],
    "vote": ["election", "result", "ballot", "tally"],
    "npp": ["new patriotic party", "akufo addo", "npp"],
    "ndc": ["national democratic congress", "mahama", "ndc"],
    "budget": ["expenditure", "revenue", "fiscal", "government spending", "ghc", "ghs"],
    "gdp": ["gross domestic product", "economic growth", "economy"],
    "revenue": ["income", "tax", "collection", "irs"],
    "expenditure": ["spending", "cost", "allocation", "disbursement"],
    "region": ["ashanti", "volta", "accra", "northern", "western", "eastern"],
    "inflation": ["price", "cpi", "cost of living", "monetary"],
    "mahama": ["ndc", "john dramani mahama", "jdm"],
    "akufo": ["npp", "nana akufo addo", "president"],
}


def _expand_query(query: str) -> str:
    lower = query.lower()
    extras = []
    for trigger, expansions in _EXPANSION_MAP.items():
        if trigger in lower:
            extras.extend(expansions)
    if extras:
        return query + " " + " ".join(set(extras))
    return query
