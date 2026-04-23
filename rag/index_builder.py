"""
Builds and caches the FAISS index from both data sources.
Run once before launching the app (or the app runs it on first start).

Student: Maame Yaa Adumaba Appiah | Index: 10022200171
Course : CS4241 - Introduction to Artificial Intelligence | ACity 2026
"""
from __future__ import annotations
import os
import pickle
import sys
from rag.data_loader import (
    load_election_csv, election_records_to_text, load_budget_pdf, clean_text,
)
from rag.chunker import (
    chunk_csv_records, chunk_by_paragraph, chunk_sliding_window, add_metadata,
)
from rag.retriever import VectorStore

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = os.path.join(BASE, "data", "vector_store.pkl")


def build_index(
    election_csv: str,
    budget_pdf: str,
    strategy: str = "paragraph",
    force_rebuild: bool = False,
) -> VectorStore:

    # ── Try loading from cache first ────────────────────────────────────────
    if not force_rebuild and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                store = pickle.load(f)
            print(f"Index loaded from cache ({store.index.ntotal} chunks)")
            return store
        except Exception as e:
            # Pickle may be incompatible (e.g. different Python/FAISS version)
            print(f"Cache load failed ({e}) — rebuilding from source …")

    # ── Validate source files exist ──────────────────────────────────────────
    if not os.path.exists(election_csv):
        raise FileNotFoundError(
            f"Election CSV not found: {election_csv}\n"
            f"Working dir: {os.getcwd()} | Python: {sys.version}"
        )
    if not os.path.exists(budget_pdf):
        raise FileNotFoundError(
            f"Budget PDF not found: {budget_pdf}\n"
            f"Working dir: {os.getcwd()} | Python: {sys.version}"
        )

    print("Building index from source files …")
    print(f"  CSV : {election_csv}")
    print(f"  PDF : {budget_pdf}")

    # ── Load & chunk election data ───────────────────────────────────────────
    records = load_election_csv(election_csv)
    election_texts = election_records_to_text(records)
    election_chunks = chunk_csv_records(election_texts)
    election_meta = add_metadata(election_chunks, source="Ghana_Election_Results")

    # ── Load & chunk budget PDF ──────────────────────────────────────────────
    raw_budget = load_budget_pdf(budget_pdf)
    clean_budget = clean_text(raw_budget)
    if strategy == "paragraph":
        budget_chunks = chunk_by_paragraph(clean_budget, max_size=800)
    else:
        budget_chunks = chunk_sliding_window(clean_budget, chunk_size=500, overlap=100)
    budget_meta = add_metadata(budget_chunks, source="Ghana_2025_Budget")
    for i, c in enumerate(budget_meta):
        c["id"] = len(election_meta) + i

    all_chunks = election_meta + budget_meta
    print(f"  Election chunks : {len(election_meta)}")
    print(f"  Budget chunks   : {len(budget_meta)}")
    print(f"  Total           : {len(all_chunks)}")

    # ── Build FAISS index ────────────────────────────────────────────────────
    store = VectorStore()
    store.build(all_chunks)

    # ── Save cache ───────────────────────────────────────────────────────────
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(store, f)
        print(f"Index cached to {CACHE_FILE}")
    except Exception as e:
        print(f"Warning: could not save cache ({e}) — continuing without cache")

    return store
