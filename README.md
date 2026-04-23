# CS4241 — Introduction to Artificial Intelligence (2026)
**RAG Chatbot: Ghana Elections + 2025 Budget Statement**  


---

## Student Information
| Field | Details |
|-------|---------|
| **Name** | Maame Yaa Adumaba Appiah |
| **Index Number** | 10022200171 |
| **Course** | CS4241 - Introduction to Artificial Intelligence |
| **Lecturer** | Godwin N. Danso |
| **Institution** | Academic City University College |
| **Year** | 2026 |

> **Live App:** *(https://cs10022200171-6mdjvbdwzy6optdzhgkrsa.streamlit.app/)*  
> **Video Walkthrough:** *(add your YouTube/Drive link here)*

---

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** chatbot without end-to-end frameworks (no LangChain, LlamaIndex). All core components — chunking, embedding, vector storage, retrieval, and prompt construction — are implemented from scratch.

## Data Sources
| Source | Description |
|--------|-------------|
| `Ghana_Election_Result.csv` | Presidential election results 1992–2020 by region |
| `2025-Budget-Statement-and-Economic-Policy_v4.pdf` | Ghana's 2025 Budget Statement |

## Architecture
```
User Query
    │
    ▼
[Query Expansion] — synonym augmentation
    │
    ▼
[FAISS Vector Store] ◄── Sentence Transformers (all-MiniLM-L6-v2)
    │
    ▼
[Hybrid Re-rank] — keyword pre-filter + vector cosine similarity
    │
    ▼
[Context Selection] — truncate to 6000 chars (~1500 tokens)
    │
    ▼
[Prompt Builder] ◄── Conversation Memory (last 6 turns)
    │
    ▼
[Claude LLM] — Haiku / Sonnet / Opus
    │
    ▼
Response + Stage Logs
```

## Part Mapping

| Part | Description | Files |
|------|-------------|-------|
| A | Data Engineering & Preparation | `rag/data_loader.py`, `rag/chunker.py` |
| B | Custom Retrieval System | `rag/embedder.py`, `rag/retriever.py` |
| C | Prompt Engineering | `rag/prompt.py` |
| D | Full RAG Pipeline | `rag/pipeline.py` |
| E | Critical Evaluation | Tab 2 in `app.py` |
| F | Architecture & System Design | This README + sidebar diagram |
| G | Innovation — Conversation Memory | `rag/memory.py` |

## Chunking Strategy
- **CSV (Elections):** Natural boundary chunking — one text block per Region × Year
- **PDF (Budget):** Paragraph-aware chunking (max 800 chars) — respects section structure
- **Sliding window** (alternative, 500 chars, 100 overlap) — available for comparison

Paragraph chunking was chosen for the budget PDF because the document uses clear section headings and numbered paragraphs. Splitting at double-newlines avoids breaking mid-sentence and keeps thematically related content together.

## Retrieval Design
- **Embeddings:** `all-MiniLM-L6-v2` (384-dim, normalised → cosine via dot-product)
- **Storage:** `faiss.IndexFlatIP` — exact inner-product search, no approximation error
- **Hybrid search:** Keyword pre-filter reduces candidate pool to semantically aligned chunks before vector re-ranking. This fixes the failure case where pure vector search returns topically related but factually unrelated chunks (e.g., "Northern Region 2016" matching "Northern Corridor budget allocation")

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter your Anthropic API key in the sidebar when the app opens.

## Experiment Logs
Stored in `logs/pipeline_log.jsonl` — one JSON line per query with:
- Timestamp
- Query text
- Retrieved chunks (source, score, snippet)
- Prompt character count
- Retrieval and generation latencies

## Innovation: Conversation Memory
`rag/memory.py` implements a sliding-window conversation buffer that keeps the last 6 turns verbatim. This enables follow-up questions ("and in which region did NDC win?") without repeating context in every query.

---
*Submitted by: **Maame Yaa Adumaba Appiah** (10022200171) to godwin.danso@acity.edu.gh*
